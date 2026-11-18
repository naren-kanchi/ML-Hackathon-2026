/** Small browser client for the PathSnap FastAPI service. */
const defaultBaseUrl = "http://127.0.0.1:8000";

export function createPathSnapApi(baseUrl = window.PATHSNAP_API_BASE_URL ?? defaultBaseUrl) {
  const base = baseUrl.replace(/\/$/, "");

  async function request(path) {
    const response = await fetch(`${base}${path}`, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail ?? `PathSnap API returned HTTP ${response.status}`);
    }
    return response.json();
  }

  return {
    listTrips: () => request("/api/v1/trips"),
    predictTrip: (tripId) => {
      if (!/^\d{8}$/.test(tripId)) throw new Error("Trip ID must contain exactly 8 digits");
      return request(`/api/v1/trips/${encodeURIComponent(tripId)}`);
    },
  };
}

export const pathsnapApi = createPathSnapApi();
