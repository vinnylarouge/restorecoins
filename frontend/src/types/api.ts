// Wire types — MUST stay in sync with backend/schemas.py and PROJECT_SPEC.md §6.
// These are binding across Phase A → B → C; only add optional fields, never
// remove or rename.

export interface Conditioning {
  type_exemplars?: string[];
  region_labels?: string;
}

export interface RestoreRequest {
  image: string;          // base64 PNG
  mask: string;           // base64 PNG, grayscale, 0=keep, 255=inpaint
  n_candidates: number;   // default 4, max 8
  seed?: number;
  conditioning?: Conditioning;
}

export interface Provenance {
  model_id: string;
  lora_id: string;
  controlnet_id?: string | null;
  seed: number;
  sampler: string;
  steps: number;
  guidance_scale: number;
  prompt?: string | null;
  timestamp: string;          // ISO 8601
  tool_version: string;       // backend git SHA
  input_image_sha256: string;
  mask_sha256: string;
}

export interface CandidateType {
  type_id: string;
  confidence: number;
  exemplar_image_url?: string;
}

export interface RestorationCandidate {
  image: string;                          // base64 PNG
  per_pixel_confidence: string;           // base64 PNG grayscale (uniform in Phase A)
  provenance: Provenance;
  candidate_types?: CandidateType[] | null;
}

export interface RestoreResponse {
  candidates: RestorationCandidate[];
}

export interface ProposeMaskResponse {
  probability_map: string;  // base64 PNG grayscale
}

export interface VersionResponse {
  tool_version: string;
  model_id: string;
  lora_id: string;
  controlnet_id?: string | null;
  mode: string;
  schema_version: string;
}
