function readResponseBody(error) {
  if (!error || typeof error !== "object") return null;
  return typeof error.responseBody === "string" ? error.responseBody : null;
}

function readErrorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  if (error && typeof error === "object" && typeof error.message === "string" && error.message) return error.message;
  return "unknown error";
}

function parseDeleteLabelError(responseBody) {
  if (!responseBody) return null;
  try {
    const body = JSON.parse(responseBody);
    const code = body?.error?.code;
    const message = body?.error?.message;
    const annotationReferences = body?.error?.details?.annotation_references;
    return {
      code,
      message,
      annotationReferences,
    };
  } catch {
    return null;
  }
}

export function formatDeleteLabelErrorMessage(labelName, error) {
  const fallbackMessage = readErrorMessage(error);
  const parsed = parseDeleteLabelError(readResponseBody(error));
  if (parsed?.code === "category_in_use" && typeof parsed.annotationReferences === "number") {
    const refs = parsed.annotationReferences;
    return `Cannot delete "${labelName}": ${refs} annotation${refs === 1 ? "" : "s"} still reference this class. Clear those annotations and submit before deleting.`;
  }
  if (parsed?.message) {
    return `Failed to delete label: ${parsed.message}`;
  }
  return `Failed to delete label: ${fallbackMessage}`;
}
