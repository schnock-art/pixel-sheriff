function shouldAllowNavigation(params) {
  const hasUnsavedDrafts = Boolean(params?.hasUnsavedDrafts);
  if (!hasUnsavedDrafts) return true;

  const confirmDiscard = params?.confirmDiscard;
  if (typeof confirmDiscard !== "function") return false;
  return Boolean(confirmDiscard());
}

module.exports = {
  shouldAllowNavigation,
};

