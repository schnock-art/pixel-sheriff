import { useCallback, useEffect, useMemo, useState } from "react";

import type { AssetSequence } from "../api";

interface UseSequenceNavigationParams {
  sequence: AssetSequence | null;
  currentAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
  pauseSignal?: number;
  thumbnailWindowSize?: number;
}

export function useSequenceNavigation({
  sequence,
  currentAssetId,
  onSelectAsset,
  pauseSignal = 0,
  thumbnailWindowSize = 7,
}: UseSequenceNavigationParams) {
  const [isPlaying, setIsPlaying] = useState(false);
  const assets = sequence?.assets ?? [];
  const totalFrames = assets.length;

  const currentIndex = useMemo(() => {
    if (!currentAssetId) return totalFrames > 0 ? 0 : -1;
    return assets.findIndex((asset) => asset.id === currentAssetId);
  }, [assets, currentAssetId, totalFrames]);

  const safeIndex = currentIndex >= 0 ? currentIndex : 0;
  const currentFrame = totalFrames > 0 ? assets[safeIndex] : null;

  const goToIndex = useCallback((index: number) => {
    if (!assets.length) return;
    const nextIndex = Math.min(Math.max(index, 0), assets.length - 1);
    onSelectAsset(assets[nextIndex].id);
  }, [assets, onSelectAsset]);

  const goToFirst = useCallback(() => goToIndex(0), [goToIndex]);
  const goToLast = useCallback(() => goToIndex(Math.max(assets.length - 1, 0)), [assets.length, goToIndex]);
  const goToPrev = useCallback(() => goToIndex(Math.max(safeIndex - 1, 0)), [goToIndex, safeIndex]);
  const goToNext = useCallback(() => goToIndex(Math.min(safeIndex + 1, Math.max(assets.length - 1, 0))), [assets.length, goToIndex, safeIndex]);
  const jumpBy = useCallback((delta: number) => goToIndex(safeIndex + delta), [goToIndex, safeIndex]);
  const pendingAssetIndices = useMemo(
    () =>
      assets
        .map((asset, index) => ({ asset, index }))
        .filter(({ asset }) => Number(asset.pending_prelabel_count ?? 0) > 0)
        .map(({ index }) => index),
    [assets],
  );
  const goToNextPending = useCallback(() => {
    if (pendingAssetIndices.length === 0) return;
    const nextIndex = pendingAssetIndices.find((index) => index > safeIndex) ?? pendingAssetIndices[0];
    goToIndex(nextIndex);
  }, [goToIndex, pendingAssetIndices, safeIndex]);

  const togglePlayback = useCallback(() => {
    if (assets.length <= 1) return;
    setIsPlaying((previous) => !previous);
  }, [assets.length]);

  const pausePlayback = useCallback(() => setIsPlaying(false), []);

  useEffect(() => {
    setIsPlaying(false);
  }, [pauseSignal, sequence?.id]);

  useEffect(() => {
    if (!isPlaying || assets.length <= 1) return;
    const playbackIntervalMs =
      sequence?.fps && Number.isFinite(sequence.fps) && sequence.fps > 0
        ? Math.max(Math.round(1000 / sequence.fps), 100)
        : 400;
    const timer = window.setInterval(() => {
      if (safeIndex >= assets.length - 1) {
        setIsPlaying(false);
        return;
      }
      onSelectAsset(assets[safeIndex + 1].id);
    }, playbackIntervalMs);
    return () => window.clearInterval(timer);
  }, [assets, isPlaying, onSelectAsset, safeIndex, sequence?.fps]);

  const thumbnailAssets = useMemo(() => {
    if (!assets.length) return [];
    const before = Math.floor(thumbnailWindowSize / 2);
    const after = thumbnailWindowSize - before - 1;
    const start = Math.max(safeIndex - before, 0);
    const end = Math.min(start + before + after + 1, assets.length);
    const adjustedStart = Math.max(end - thumbnailWindowSize, 0);
    return assets.slice(adjustedStart, end);
  }, [assets, safeIndex, thumbnailWindowSize]);

  return {
    assets,
    totalFrames,
    currentIndex,
    currentFrame,
    isPlaying,
    goToIndex,
    goToFirst,
    goToLast,
    goToPrev,
    goToNext,
    jumpBy,
    goToNextPending,
    togglePlayback,
    pausePlayback,
    thumbnailAssets,
    pendingFrameCount: pendingAssetIndices.length,
  };
}
