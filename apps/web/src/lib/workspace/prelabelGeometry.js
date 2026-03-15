function resolvePrelabelBBox(proposal) {
  if (Array.isArray(proposal?.reviewed_bbox) && proposal.reviewed_bbox.length === 4) {
    return proposal.reviewed_bbox;
  }
  return Array.isArray(proposal?.bbox) ? proposal.bbox : [];
}

function resolvePrelabelCategoryId(proposal) {
  if (typeof proposal?.reviewed_category_id === "string" && proposal.reviewed_category_id.trim() !== "") {
    return proposal.reviewed_category_id;
  }
  return typeof proposal?.category_id === "string" ? proposal.category_id : "";
}

module.exports = {
  resolvePrelabelBBox,
  resolvePrelabelCategoryId,
};
