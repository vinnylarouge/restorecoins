import { useReducer } from "react";
import type { RestorationCandidate } from "../types/api";

// All session state lives in memory; nothing is persisted. Spec §4.6:
// "No login, no accounts. Stateless frontend: all state lives in URL params or
//  browser session; no backend database in Phase A."
export type Stage = "upload" | "mask" | "results";

export interface SessionState {
  stage: Stage;
  imageB64?: string;           // input coin as PNG b64
  imageDataUrl?: string;       // for display
  maskB64?: string;            // base64 grayscale PNG (0=keep, 255=inpaint)
  maskDataUrl?: string;
  proposedMaskB64?: string;    // server-suggested probability map
  candidates: RestorationCandidate[];
  selectedCandidate: number;
  loading: false | "propose" | "restore";
  error?: string;
}

export type Action =
  | { type: "image_loaded"; b64: string; dataUrl: string }
  | { type: "back_to_upload" }
  | { type: "back_to_mask" }
  | { type: "proposed_mask"; b64: string }
  | { type: "user_mask"; b64: string; dataUrl: string }
  | { type: "loading"; what: SessionState["loading"] }
  | { type: "candidates"; candidates: RestorationCandidate[] }
  | { type: "select_candidate"; index: number }
  | { type: "error"; message: string }
  | { type: "clear_error" };

const initial: SessionState = {
  stage: "upload",
  candidates: [],
  selectedCandidate: 0,
  loading: false,
};

function reducer(s: SessionState, a: Action): SessionState {
  switch (a.type) {
    case "image_loaded":
      return { ...initial, stage: "mask", imageB64: a.b64, imageDataUrl: a.dataUrl };
    case "back_to_upload":
      return initial;
    case "back_to_mask":
      return { ...s, stage: "mask", candidates: [], selectedCandidate: 0 };
    case "proposed_mask":
      return { ...s, proposedMaskB64: a.b64 };
    case "user_mask":
      return { ...s, maskB64: a.b64, maskDataUrl: a.dataUrl };
    case "loading":
      return { ...s, loading: a.what, error: undefined };
    case "candidates":
      return { ...s, stage: "results", candidates: a.candidates,
               selectedCandidate: 0, loading: false };
    case "select_candidate":
      return { ...s, selectedCandidate: a.index };
    case "error":
      return { ...s, error: a.message, loading: false };
    case "clear_error":
      return { ...s, error: undefined };
  }
}

export function useSession() {
  return useReducer(reducer, initial);
}
