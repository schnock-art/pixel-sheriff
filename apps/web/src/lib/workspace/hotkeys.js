function parseLabelShortcutDigit(event) {
  const digitCodeMatch = event.code.match(/^(Digit|Numpad)([1-9])$/);
  if (digitCodeMatch) return Number(digitCodeMatch[2]);
  if (/^[1-9]$/.test(event.key)) return Number(event.key);
  return null;
}

function shouldIgnoreKeyboardTarget(target) {
  const tag = target?.tagName?.toLowerCase?.();
  return tag === "input" || tag === "textarea" || Boolean(target?.isContentEditable);
}

function resolveWorkspaceHotkeyAction(event, context) {
  if (shouldIgnoreKeyboardTarget(event.target)) return null;
  if (event.altKey || event.ctrlKey || event.metaKey) return null;

  if (event.key === "ArrowLeft") return { type: "navigate_prev" };
  if (event.key === "ArrowRight") return { type: "navigate_next" };

  const digit = parseLabelShortcutDigit(event);
  if (digit === null || digit > context.activeLabelCount) return null;
  return { type: "toggle_label", labelIndex: digit - 1 };
}

module.exports = {
  parseLabelShortcutDigit,
  shouldIgnoreKeyboardTarget,
  resolveWorkspaceHotkeyAction,
};
