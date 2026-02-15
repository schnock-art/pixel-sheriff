export interface HotkeyKeyboardTarget {
  tagName?: string;
  isContentEditable?: boolean;
}

export interface WorkspaceHotkeyEvent {
  key: string;
  code: string;
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  target?: unknown;
}

export type WorkspaceHotkeyAction =
  | { type: "navigate_prev" }
  | { type: "navigate_next" }
  | { type: "toggle_label"; labelIndex: number };

export function parseLabelShortcutDigit(event: WorkspaceHotkeyEvent): number | null;
export function shouldIgnoreKeyboardTarget(target: unknown): boolean;
export function resolveWorkspaceHotkeyAction(
  event: WorkspaceHotkeyEvent,
  context: { activeLabelCount: number },
): WorkspaceHotkeyAction | null;
