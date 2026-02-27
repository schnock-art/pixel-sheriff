export interface ClassColor {
  hue: number;
  chipBackground: string;
  chipBorder: string;
  chipText: string;
  chipActiveBackground: string;
  overlayStroke: string;
  overlayFill: string;
}

export function normalizedHueForClassId(classId: number): number;
export function getClassColor(classId: number): ClassColor;
