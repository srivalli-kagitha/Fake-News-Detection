import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 0 });

export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  if (err?.code === "ERR_NETWORK") return "Cannot reach the backend. Is it running on port 8000?";
  return err?.message || "Something went wrong.";
}

export const getHealth = () => api.get("/health").then((r) => r.data);

export const analyzeText = (payload) => api.post("/text/analyze", payload).then((r) => r.data);

export const explainText = (payload) => api.post("/text/explain", payload).then((r) => r.data);

export const verifyClaims = (payload) => api.post("/claims/verify", payload).then((r) => r.data);

export function analyzeImage(file, explain = true) {
  const form = new FormData();
  form.append("file", file);
  form.append("explain", explain);
  return api.post("/image/analyze", form).then((r) => r.data);
}

export function batchText(file, language, verifyClaims) {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  form.append("verify_claims", verifyClaims);
  return api.post("/batch/text", form).then((r) => r.data);
}

export function batchImages(files) {
  const form = new FormData();
  [...files].forEach((f) => form.append("files", f));
  return api.post("/batch/images", form).then((r) => r.data);
}

export const downloadUrl = (jobId) => `/api/batch/${jobId}/download`;

export const getHistory = (limit = 100) => api.get("/history", { params: { limit } }).then((r) => r.data);
export const deleteHistory = (id) => api.delete(`/history/${id}`).then((r) => r.data);
export const clearHistory = () => api.delete("/history").then((r) => r.data);
