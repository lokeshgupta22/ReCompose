export async function analyze(file, { aspects, topK } = {}) {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams();
  if (aspects?.length) params.set("aspects", aspects.join(","));
  if (topK) params.set("top_k", String(topK));
  const query = params.toString();
  const response = await fetch(`/api/analyze${query ? `?${query}` : ""}`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Analysis failed (HTTP ${response.status})`);
  }
  return response.json();
}
