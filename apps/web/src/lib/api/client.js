const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

class ApiError extends Error {
  constructor(params) {
    super(params.message);
    this.name = "ApiError";
    this.url = params.url;
    this.method = params.method;
    this.status = params.status;
    this.responseBody = params.responseBody;
  }
}

function getApiBase() {
  return API_BASE;
}

async function requestJson(path, init) {
  const url = `${API_BASE}/api/v1${path}`;
  const method = (init.method ?? "GET").toUpperCase();

  let response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new ApiError({
      message: `NetworkError on ${method} ${url}`,
      url,
      method,
      responseBody: error instanceof Error ? error.message : String(error),
    });
  }

  if (!response.ok) {
    const responseBody = await response.text();
    throw new ApiError({
      message: `Request failed (${response.status}) on ${method} ${url}`,
      url,
      method,
      status: response.status,
      responseBody,
    });
  }

  return response.json();
}

async function requestNoContent(path, init) {
  const url = `${API_BASE}/api/v1${path}`;
  const method = (init.method ?? "GET").toUpperCase();

  let response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new ApiError({
      message: `NetworkError on ${method} ${url}`,
      url,
      method,
      responseBody: error instanceof Error ? error.message : String(error),
    });
  }

  if (!response.ok) {
    const responseBody = await response.text();
    throw new ApiError({
      message: `Request failed (${response.status}) on ${method} ${url}`,
      url,
      method,
      status: response.status,
      responseBody,
    });
  }
}

async function apiGet(path) {
  return requestJson(path, { cache: "no-store" });
}

async function apiPost(path, body) {
  return requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function apiPostForm(path, formData) {
  return requestJson(path, {
    method: "POST",
    body: formData,
  });
}

function resolveAssetUri(uri) {
  if (uri.startsWith("http://") || uri.startsWith("https://") || uri.startsWith("blob:") || uri.startsWith("data:")) {
    return uri;
  }
  if (uri.startsWith("/")) {
    return `${API_BASE}${uri}`;
  }
  return `${API_BASE}/${uri}`;
}

module.exports = {
  ApiError,
  getApiBase,
  requestJson,
  requestNoContent,
  apiGet,
  apiPost,
  apiPostForm,
  resolveAssetUri,
};
