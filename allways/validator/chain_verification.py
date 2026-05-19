"""Verifies both sides of a swap using chain providers and on-chain swap data."""

import asyncio
from typing import Dict, Optional, Set

import bittensor as bt

from allways.chain_providers.base import ChainProvider, ProviderUnreachableError, TransactionInfo
from allways.classes import Swap
from allways.utils.logging import log_on_change
from allways.utils.rate import expected_swap_amounts

# Subtract one block from the observed dest-chain tip so a tx mined in the
# exact block we sampled still passes. Replay txs are hundreds of blocks
# old, so this grace doesn't weaken the defense.
DEST_TIP_SNAPSHOT_GRACE = 1


class SwapVerifier:
    """Verifies swap transactions on both source and destination chains.

    Rate and miner source address are stored on the swap struct at initiation,
    so verification is self-contained — no commitment lookup needed.

    Dest-tx replay defense: on first observation of a swap, the validator
    snapshots the dest chain's current tip. A miner-supplied dest tx must
    be mined at or after that snapshot, otherwise it's a recycled hash from
    an older swap. A contract-level ``used_to_tx`` mirror of ``used_from_tx``
    would be stronger but requires a contract upgrade; this validator-side
    check ships the same protection without one.
    """

    def __init__(
        self,
        chain_providers: Dict[str, ChainProvider],
        fee_divisor: int = 100,
    ):
        self.providers = chain_providers
        self.fee_divisor = fee_divisor
        self.last_logged_confs: Dict[str, int] = {}  # swap_id:chain -> confs
        self.source_verified_ids: Set[int] = set()  # source tx is final once confirmed
        # swap_id -> dest-chain block height observed at first sighting.
        # Absent = no snapshot yet (fail-open with a one-time warning).
        # TAO-dest swaps don't go in here — they use swap.initiated_block directly.
        self.dest_tip_at_init: Dict[int, int] = {}

    def observe_initiation(self, swap: Swap) -> None:
        """Snapshot the dest chain's tip the first time a non-TAO-dest swap appears.

        Idempotent. Silently retries next forward step on transient provider
        failure (single warning emitted via ``log_on_change``).
        """
        if swap.to_chain == 'tao' or swap.id in self.dest_tip_at_init:
            return
        provider = self.providers.get(swap.to_chain)
        if provider is None:
            return
        try:
            tip = provider.get_current_block_height()
        except Exception as e:
            tip = None
            bt.logging.debug(f'Swap {swap.id}: dest-tip snapshot raised {type(e).__name__}: {e}')
        if tip is None or tip <= 0:
            log_on_change(
                f'snapshot_unavailable:{swap.id}',
                True,
                f'Swap {swap.id}: dest-tip snapshot unavailable on {swap.to_chain} — '
                f'replay defense disabled until snapshot succeeds',
            )
            return
        self.dest_tip_at_init[swap.id] = tip - DEST_TIP_SNAPSHOT_GRACE

    def prune_to_active(self, active_ids: Set[int]) -> None:
        """Drop per-swap state for swaps no longer being tracked."""
        self.dest_tip_at_init = {sid: v for sid, v in self.dest_tip_at_init.items() if sid in active_ids}
        self.source_verified_ids &= active_ids

    def verify_tx(
        self,
        swap: Swap,
        chain: str,
        tx_hash: str,
        expected_recipient: str,
        expected_amount: int,
        block_hint: int = 0,
        expected_sender: str = '',
    ) -> Optional[TransactionInfo]:
        """Verify a confirmed transaction on a specific chain.

        Defers tx lookup, amount, and sender checks to the provider's
        ``verify_transaction`` so the defense lives in one place shared with
        the miner and axon flows. Keeps the rate-limited confirmations debug
        log here because it's specific to the validator polling loop.
        """
        provider = self.providers.get(chain)
        if not provider:
            bt.logging.warning(f'Swap {swap.id}: no provider for chain {chain}')
            return None

        if not tx_hash:
            bt.logging.debug(f'Swap {swap.id}: empty tx_hash for {chain}, skipping verification')
            return None

        try:
            tx_info = provider.verify_transaction(
                tx_hash=tx_hash,
                expected_recipient=expected_recipient,
                expected_amount=expected_amount,
                block_hint=block_hint,
                expected_sender=expected_sender or None,
            )
            if tx_info is None:
                bt.logging.debug(
                    f'Swap {swap.id}: verify_transaction returned None on {chain} '
                    f'(tx={tx_hash[:16]}... block_hint={block_hint})'
                )
                return None
            if not tx_info.confirmed:
                self.log_confs_progress(swap.id, chain, tx_hash, tx_info, expected_recipient, expected_amount)
                return None
            return tx_info
        except ProviderUnreachableError:
            raise
        except Exception as e:
            bt.logging.error(f'Swap {swap.id}: verification error on {chain}: {e}')
            return None

    def is_dest_tx_fresh(self, swap: Swap, dest_info: TransactionInfo) -> bool:
        """Reject a dest tx mined before the swap was initiated (replay defense)."""
        if dest_info.block_number is None:
            return True
        lower = swap.initiated_block if swap.to_chain == 'tao' else self.dest_tip_at_init.get(swap.id)
        if lower is None:
            log_on_change(
                f'snapshot_failopen:{swap.id}',
                True,
                f'Swap {swap.id}: no dest-tip snapshot — replay defense skipped (fail-open)',
            )
            return True
        if dest_info.block_number < lower:
            bt.logging.warning(
                f'Swap {swap.id}: dest tx at block {dest_info.block_number} < initiated {lower} — '
                f'rejecting as replay (tx={swap.to_tx_hash[:16]}...)'
            )
            return False
        return True

    def log_confs_progress(
        self,
        swap_id: int,
        chain: str,
        tx_hash: str,
        tx_info: TransactionInfo,
        expected_recipient: str,
        expected_amount: int,
    ) -> None:
        """Rate-limited debug log for confirmations progress on unconfirmed txs."""
        log_key = f'{swap_id}:{chain}'
        if self.last_logged_confs.get(log_key) == tx_info.confirmations:
            return
        self.last_logged_confs[log_key] = tx_info.confirmations
        bt.logging.debug(
            f'Swap {swap_id}: tx found but not confirmed on {chain} '
            f'(confs={tx_info.confirmations} tx={tx_hash[:16]}... '
            f'addr={expected_recipient[:16]}... expected={expected_amount})'
        )

    async def verify_miner_fulfillment(self, swap: Swap) -> bool:
        """Verify rate, to_amount, user send, and miner fulfillment.

        Rate and miner source address are read directly from the swap struct
        (snapshotted at initiation), so this works regardless of miner registration.
        """
        if not swap.rate or not swap.miner_from_address:
            bt.logging.warning(f'Swap {swap.id}: missing rate or miner_from_address on swap struct')
            return False

        _, expected_user_receives = expected_swap_amounts(swap, self.fee_divisor)
        if expected_user_receives == 0:
            bt.logging.warning(f'Swap {swap.id}: rate produces 0 to_amount after fees')
            return False

        if int(swap.to_amount) != expected_user_receives:
            bt.logging.warning(
                f'Swap {swap.id}: to_amount mismatch — expected {expected_user_receives}, contract has {swap.to_amount}'
            )
            return False

        # Sequential — parallel threads contend on the shared substrate WS.
        if swap.id in self.source_verified_ids:
            source_ok = True
        else:
            source_info = await asyncio.to_thread(
                self.verify_tx,
                swap,
                swap.from_chain,
                swap.from_tx_hash,
                swap.miner_from_address,
                swap.from_amount,
                swap.from_tx_block,
            )
            source_ok = source_info is not None
            if source_ok:
                self.source_verified_ids.add(swap.id)

        dest_info = await asyncio.to_thread(
            self.verify_tx,
            swap,
            swap.to_chain,
            swap.to_tx_hash,
            swap.user_to_address,
            expected_user_receives,
            swap.to_tx_block,
            swap.miner_to_address,
        )
        dest_ok = dest_info is not None and self.is_dest_tx_fresh(swap, dest_info)

        if source_ok != dest_ok:
            log_on_change(
                f'partial:{swap.id}',
                (source_ok, dest_ok),
                f'Swap {swap.id}: partial verification (source={source_ok}, dest={dest_ok}) — '
                f'{"dest" if source_ok else "source"} side blocking confirm',
            )

        return source_ok and dest_ok
