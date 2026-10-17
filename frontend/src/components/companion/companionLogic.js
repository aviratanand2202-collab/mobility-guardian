/**
 * Pure Accompaniment & Cohesion Logic for Trusted Companion Mode.
 *
 * Requirements:
 * 1. Calculate distance between Patient and consenting Companion.
 * 2. Accompaniment Status:
 *    - "With Companion": Both locations fresh and distance <= threshold.
 *    - "Separated": Both locations fresh and distance > threshold.
 *    - "Accompaniment Unverified": Location is missing, stale (>60s), or unavailable.
 * 3. Safe Zone Cohesion:
 *    - If Patient and Companion leave the safe zone together, do NOT trigger
 *      a group-separation alert just because they left the zone.
 * 4. Privacy:
 *    - Companion personal ID is never exposed; privacy alias is used.
 */
export const EARTH_RADIUS_METERS = 6371000

export function haversineMeters(coord1, coord2) {
  if (!coord1 || !coord2) return 0
  const [lat1, lon1] = coord1
  const [lat2, lon2] = coord2

  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2)

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return Math.round(EARTH_RADIUS_METERS * c * 10) / 10
}

export const DEFAULT_GEOFENCE = {
  name: 'Default Safe Zone',
  center: [37.7735, -122.4215],
  radiusMeters: 260,
}

export function isInsideGeofence(coord, geofence = DEFAULT_GEOFENCE) {
  if (!coord || !geofence) return true
  const center = geofence.center
  const radius = geofence.radius_m || geofence.radiusMeters || 260
  if (!center) return true
  const dist = haversineMeters(coord, center)
  return dist <= radius
}


export const ACCOMPANIMENT_STATUS = {
  WITH_COMPANION: 'With Companion',
  SEPARATED: 'Separated',
  ACCOMPANIMENT_UNVERIFIED: 'Accompaniment Unverified',
}

/**
 * Evaluates real-time accompaniment status between Patient and Companion.
 *
 * @param {Object} params
 * @param {Object|null} params.patientLocation - { lat, lng } or null
 * @param {Object|null} params.companionLocation - { lat, lng } or null
 * @param {string|number|Date|null} params.companionTimestamp - timestamp of last companion update
 * @param {number} [params.thresholdMeters=50] - maximum distance to be considered "With Companion"
 * @param {number} [params.maxStaleSeconds=60] - maximum allowable age for fresh companion telemetry
 * @param {number} [params.currentTime=Date.now()] - current timestamp in ms
 * @param {Object} [params.geofence=DEFAULT_GEOFENCE] - safe zone geofence definition
 */
export function evaluateAccompaniment({
  patientLocation,
  companionLocation,
  companionTimestamp,
  thresholdMeters = 50,
  maxStaleSeconds = 60,
  currentTime = Date.now(),
  geofence = DEFAULT_GEOFENCE,
}) {
  // 1. Check patient coordinates availability
  if (!patientLocation || patientLocation.lat == null || patientLocation.lng == null) {
    return {
      status: ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED,
      isVerified: false,
      reason: 'Patient location is unavailable',
      distanceMeters: null,
      isStale: true,
      ageSeconds: null,
      safeZoneDepartureAccompanied: false,
      statusMessage: 'Accompaniment unverified · Awaiting patient satellite fix',
    }
  }

  // 2. Check companion coordinates availability
  if (!companionLocation || companionLocation.lat == null || companionLocation.lng == null) {
    return {
      status: ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED,
      isVerified: false,
      reason: 'Companion location has not been reported',
      distanceMeters: null,
      isStale: true,
      ageSeconds: null,
      safeZoneDepartureAccompanied: false,
      statusMessage: 'Accompaniment unverified · Awaiting companion device fix',
    }
  }

  // 3. Check companion freshness
  let ageSeconds = null
  if (companionTimestamp) {
    const tsMs = new Date(companionTimestamp).getTime()
    if (!isNaN(tsMs)) {
      ageSeconds = Math.max(0, Math.floor((currentTime - tsMs) / 1000))
    }
  }

  const isStale = ageSeconds == null || ageSeconds > maxStaleSeconds
  if (isStale) {
    return {
      status: ACCOMPANIMENT_STATUS.ACCOMPANIMENT_UNVERIFIED,
      isVerified: false,
      reason: `Companion location is stale (${ageSeconds != null ? `${ageSeconds}s old` : 'unknown age'})`,
      distanceMeters: null,
      isStale: true,
      ageSeconds,
      safeZoneDepartureAccompanied: false,
      statusMessage: `Accompaniment unverified · Companion location is stale (${ageSeconds != null ? `${ageSeconds}s ago` : 'delayed'})`,
    }
  }

  // 4. Calculate actual Haversine distance
  const pCoords = [patientLocation.lat, patientLocation.lng]
  const cCoords = [companionLocation.lat, companionLocation.lng]
  const distanceMeters = haversineMeters(pCoords, cCoords)

  // 5. Check Geofence Safe Zone boundaries
  const patientInside = isInsideGeofence(pCoords, geofence)
  const companionInside = isInsideGeofence(cCoords, geofence)
  const outsideZoneTogether = !patientInside && !companionInside && distanceMeters <= thresholdMeters

  if (distanceMeters <= thresholdMeters) {
    const msg = outsideZoneTogether
      ? `Group travelling together outside zone (Accompanied — no emergency alert, separation: ${Math.round(distanceMeters)}m)`
      : `With companion · Near friend (separation: ${Math.round(distanceMeters)}m, threshold: ${thresholdMeters}m)`

    return {
      status: ACCOMPANIMENT_STATUS.WITH_COMPANION,
      isVerified: true,
      reason: null,
      distanceMeters,
      isStale: false,
      ageSeconds,
      safeZoneDepartureAccompanied: outsideZoneTogether,
      statusMessage: msg,
    }
  }

  // 6. Separated beyond threshold
  return {
    status: ACCOMPANIMENT_STATUS.SEPARATED,
    isVerified: true,
    reason: `Patient and companion separated by ${Math.round(distanceMeters)}m (threshold: ${thresholdMeters}m)`,
    distanceMeters,
    isStale: false,
    ageSeconds,
    safeZoneDepartureAccompanied: false,
    statusMessage: `Separated from companion · Distance: ${Math.round(distanceMeters)}m (Threshold: ${thresholdMeters}m)`,
  }
}
