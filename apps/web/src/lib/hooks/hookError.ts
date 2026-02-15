export type HookError = {
  message: string;
  detail: string | null;
};

export function toHookError(error: unknown, message: string): HookError {
  if (error instanceof Error) {
    return {
      message,
      detail: error.message || null,
    };
  }

  if (typeof error === "string" && error.trim() !== "") {
    return {
      message,
      detail: error,
    };
  }

  return {
    message,
    detail: null,
  };
}
