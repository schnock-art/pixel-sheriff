import { useCallback, useEffect, useMemo, useState } from "react";

import {
  acceptPrelabelProposals,
  getPrelabelSession,
  listPrelabelProposals,
  rejectPrelabelProposals,
  type PrelabelProposal,
  type PrelabelSession,
} from "../api";
import { resolvePrelabelBBox, resolvePrelabelCategoryId } from "../workspace/prelabelGeometry.js";
import { toHookError, type HookError } from "./hookError";


interface GeometryBBoxObject {
  id: string;
  kind: "bbox";
  category_id: string;
  bbox: number[];
  provenance?: {
    origin_kind: string;
    session_id?: string;
    proposal_id?: string;
    source_model?: string;
    prompt_text?: string;
    confidence?: number;
    review_decision?: string;
  };
}

interface GeometryPolygonObject {
  id: string;
  kind: "polygon";
  category_id: string;
  segmentation: number[][];
  provenance?: GeometryBBoxObject["provenance"];
}

type GeometryObject = GeometryBBoxObject | GeometryPolygonObject;


export function usePrelabels({
  projectId,
  taskId,
  sessionId,
  sessionStatus,
  currentAssetId,
  currentObjects,
  onLoadProposalIntoDraft,
  onRefresh,
  setMessage,
}: {
  projectId: string | null;
  taskId: string | null;
  sessionId: string | null;
  sessionStatus: string | null;
  currentAssetId: string | null;
  currentObjects: GeometryObject[];
  onLoadProposalIntoDraft: (objects: GeometryObject[]) => void;
  onRefresh: () => Promise<void>;
  setMessage: (message: string | null) => void;
}) {
  const [session, setSession] = useState<PrelabelSession | null>(null);
  const [proposals, setProposals] = useState<PrelabelProposal[]>([]);
  const [selectedProposalId, setSelectedProposalId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<HookError | null>(null);

  const load = useCallback(
    async (isActive: () => boolean = () => true) => {
      if (!projectId || !taskId || !sessionId) {
        if (!isActive()) return;
        setSession(null);
        setProposals([]);
        setError(null);
        setIsLoading(false);
        return;
      }
      try {
        if (isActive()) {
          setIsLoading(true);
          setError(null);
        }
        const [sessionResponse, proposalsResponse] = await Promise.all([
          getPrelabelSession(projectId, taskId, sessionId),
          listPrelabelProposals(projectId, taskId, sessionId, {
            asset_id: currentAssetId,
            status_filter: "pending",
          }),
        ]);
        if (!isActive()) return;
        setSession(sessionResponse.session);
        setProposals(proposalsResponse.items);
      } catch (err) {
        if (!isActive()) return;
        setError(toHookError(err, "Failed to load AI prelabels"));
        setSession(null);
        setProposals([]);
      } finally {
        if (isActive()) setIsLoading(false);
      }
    },
    [currentAssetId, projectId, sessionId, taskId],
  );

  useEffect(() => {
    let isActive = true;
    void load(() => isActive);
    return () => {
      isActive = false;
    };
  }, [load]);

  useEffect(() => {
    if (!projectId || !taskId || !sessionId) return;
    if (!["queued", "running"].includes(String(session?.status ?? sessionStatus ?? ""))) return;
    const intervalId = window.setInterval(() => {
      void load();
      void onRefresh();
    }, 2000);
    return () => window.clearInterval(intervalId);
  }, [load, onRefresh, projectId, session?.status, sessionId, sessionStatus, taskId]);

  useEffect(() => {
    if (!selectedProposalId) return;
    if (!proposals.some((proposal) => proposal.id === selectedProposalId)) {
      setSelectedProposalId(null);
    }
  }, [proposals, selectedProposalId]);

  const selectedProposal = useMemo(
    () => proposals.find((proposal) => proposal.id === selectedProposalId) ?? null,
    [proposals, selectedProposalId],
  );
  const pendingCount = proposals.length;

  const applyReviewAction = useCallback(
    async (action: "accept" | "reject", payload?: { asset_id?: string | null; proposal_ids?: string[] }) => {
      if (!projectId || !taskId || !sessionId) return;
      try {
        setIsApplying(true);
        if (action === "accept") {
          await acceptPrelabelProposals(projectId, taskId, sessionId, payload);
        } else {
          await rejectPrelabelProposals(projectId, taskId, sessionId, payload);
        }
        await Promise.all([load(), onRefresh()]);
      } catch (err) {
        setMessage(err instanceof Error ? err.message : "AI prelabel action failed.");
      } finally {
        setIsApplying(false);
      }
    },
    [load, onRefresh, projectId, sessionId, setMessage, taskId],
  );

  function editSelectedProposal() {
    if (!selectedProposal || !session) return;
    const bbox = resolvePrelabelBBox(selectedProposal);
    const categoryId = resolvePrelabelCategoryId(selectedProposal);
    const nextObject: GeometryObject = {
      id: selectedProposal.promoted_object_id ?? `prelabel-${selectedProposal.id}`,
      kind: "bbox",
      category_id: categoryId,
      bbox,
      provenance: {
        origin_kind: "ai_prelabel",
        session_id: selectedProposal.session_id,
        proposal_id: selectedProposal.id,
        source_model: session.source_ref ?? session.source_type,
        prompt_text: selectedProposal.prompt_text ?? selectedProposal.label_text,
        confidence: selectedProposal.confidence,
        review_decision: "edited",
      },
    };
    const withoutExisting = currentObjects.filter(
      (objectValue) => objectValue.provenance?.proposal_id !== selectedProposal.id,
    );
    onLoadProposalIntoDraft([...withoutExisting, nextObject]);
    setMessage(`Loaded AI proposal "${selectedProposal.label_text}" into the annotation draft.`);
  }

  return {
    session,
    proposals,
    pendingCount,
    selectedProposal,
    selectedProposalId,
    setSelectedProposalId,
    isLoading,
    isApplying,
    error,
    reload: () => load(),
    acceptSelectedProposal: () => applyReviewAction("accept", selectedProposal ? { proposal_ids: [selectedProposal.id] } : undefined),
    rejectSelectedProposal: () => applyReviewAction("reject", selectedProposal ? { proposal_ids: [selectedProposal.id] } : undefined),
    acceptCurrentFrame: () => applyReviewAction("accept", currentAssetId ? { asset_id: currentAssetId } : undefined),
    rejectCurrentFrame: () => applyReviewAction("reject", currentAssetId ? { asset_id: currentAssetId } : undefined),
    acceptFullSession: () => applyReviewAction("accept"),
    rejectFullSession: () => applyReviewAction("reject"),
    editSelectedProposal,
  };
}
