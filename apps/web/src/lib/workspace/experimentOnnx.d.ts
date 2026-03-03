export function onnxStatusLabel(
  onnxInfo: { status?: string } | null | undefined,
  experimentStatus: string | null | undefined,
): string;

export function onnxInputShapeText(
  onnxInfo: { input_shape?: Array<number | null | undefined> } | null | undefined,
): string;

export function onnxClassNamesText(
  onnxInfo: { class_names?: Array<string | null | undefined> } | null | undefined,
  options?: { maxNames?: number },
): string;

export function onnxValidationText(
  onnxInfo: { validation?: { status?: string } | null } | null | undefined,
): string;
