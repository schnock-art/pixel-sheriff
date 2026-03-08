export class ApiError extends Error {
  readonly url: string;
  readonly method: string;
  readonly status?: number;
  readonly responseBody?: string;

  constructor(params: { message: string; url: string; method: string; status?: number; responseBody?: string });
}

export function getApiBase(): string;

export function requestJson<T>(path: string, init: RequestInit): Promise<T>;

export function requestNoContent(path: string, init: RequestInit): Promise<void>;

export function apiGet<T>(path: string): Promise<T>;

export function apiPost<T, TBody = unknown>(path: string, body: TBody): Promise<T>;

export function apiPostForm<T>(path: string, formData: FormData): Promise<T>;

export function resolveAssetUri(uri: string): string;
