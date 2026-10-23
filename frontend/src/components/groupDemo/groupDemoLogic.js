/**
 * Pure logic and synthetic route coordinates for Frontend Group Demo.
 *
 * Methodological & Safety Guarantees:
 * 1. Operates 100% on synthetic demo coordinates — does NOT fabricate real GPS,
 *    telemetry, ML inference, or patient clinical risk.
 * 2. Deterministic animation steps bounded to predefined neighborhood routes.
 * 3. Safe Area (Geofence) cohesion logic:
 *    - Leaving geofence TOGETHER as a group is informational, NOT an emergency alert.
 *    - Patient separation beyond threshold for sustained duration raises a Warning,
 *      then escalates to an Alert.
 *    - Rejoining resolves/clears the alert.
 */

// Earth's mean radius in meters for Haversine distance
export const EARTH_RADIUS_METERS = 6371000

/**
 * Calculate distance in meters between two [latitude, longitude] pairs.
 */
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
  return R_round(EARTH_RADIUS_METERS * c)
}

function R_round(val) {
  return Math.round(val * 10) / 10
}

/**
 * Canonical Safe Zone (Geofence) for the demo.
 * Centered in the neighborhood park area of San Francisco coordinates.
 */
export const DEFAULT_GEOFENCE = {
  name: 'Neighborhood Safe Zone (Geofence)',
  center: [37.7735, -122.4215],
  radiusMeters: 260, // 260 meter radius safe zone
}

/**
 * Check whether a coordinate falls inside the configured circular safe zone.
 */
export function isInsideGeofence(coord, geofence = DEFAULT_GEOFENCE) {
  if (!coord || !geofence || !geofence.center) return true
  const dist = haversineMeters(coord, geofence.center)
  return dist <= geofence.radiusMeters
}

/**
 * Predefined, deterministic multi-step waypoints for the 3 demo users.
 *
 * Sequence of scenarios across the 20 steps:
 * - Steps 0 to 4: Inside Safe Zone, moving together (distance ~15-25m, cohesive).
 * - Steps 5 to 9: Departing Safe Zone TOGETHER (distance ~20-30m < threshold).
 *                 Status: "Group travelling together outside zone" (NO emergency alert).
 * - Steps 10 to 14: Patient (User 2) lingers/diverts; Companions (User 1 & 3) continue forward.
 *                   Distance exceeds threshold (65m -> 85m -> 115m -> 140m).
 *                   Step 11: Sustained >= 2 ticks -> "Group Separation Warning".
 *                   Step 13: Sustained >= 4 ticks -> Escalates to "Group Separation Alert".
 * - Steps 15 to 18: Companions notice and turn back to rejoin Patient.
 *                   Distance drops back (70m -> 35m -> 18m <= threshold).
 *                   Alert clears / resolves ("Alert Resolved — Patient Rejoined Group").
 * - Steps 19: Group returns back towards Safe Zone.
 */
export const DEMO_ROUTES = {
  sim_user_1: {
    id: 'sim_user_1',
    label: 'Simulated User 1',
    defaultRole: 'companion',
    badge: '1',
    color: '#2563eb', // Royal Blue
    dashArray: null,
    waypoints: [
      [37.7720, -122.4225], // 0: inside zone
      [37.7725, -122.4218], // 1: inside zone
      [37.7731, -122.4210], // 2: inside zone
      [37.7738, -122.4202], // 3: near zone boundary
      [37.7744, -122.4194], // 4: crossing boundary together
      [37.7750, -122.4187], // 5: outside zone together
      [37.7756, -122.4180], // 6: outside zone together
      [37.7762, -122.4173], // 7: outside zone together
      [37.7768, -122.4166], // 8: outside zone together
      [37.7774, -122.4159], // 9: outside zone together
      [37.7781, -122.4151], // 10: companions advance (patient slows)
      [37.7787, -122.4143], // 11: companions advance (warning triggered)
      [37.7793, -122.4136], // 12: companions advance
      [37.7798, -122.4130], // 13: companions advance (alert escalated)
      [37.7801, -122.4125], // 14: companions halt/turn back
      [37.7793, -122.4135], // 15: companions returning to patient
      [37.7784, -122.4146], // 16: companions closer
      [37.7776, -122.4156], // 17: companions rejoin patient
      [37.7770, -122.4163], // 18: reunited outside zone
      [37.7755, -122.4180], // 19: walking back together towards zone
    ],
  },
  sim_user_2: {
    id: 'sim_user_2',
    label: 'Simulated User 2',
    defaultRole: 'patient', // Selected Patient by default
    badge: '2',
    color: '#059669', // Emerald Green
    dashArray: '6, 6',
    waypoints: [
      [37.7719, -122.4223], // 0: inside zone with companion
      [37.7724, -122.4216], // 1: inside zone
      [37.7730, -122.4208], // 2: inside zone
      [37.7737, -122.4200], // 3: near zone boundary
      [37.7743, -122.4192], // 4: crossing boundary together
      [37.7749, -122.4185], // 5: outside zone together
      [37.7755, -122.4178], // 6: outside zone together
      [37.7761, -122.4171], // 7: outside zone together
      [37.7767, -122.4164], // 8: outside zone together
      [37.7772, -122.4158], // 9: outside zone together
      [37.7773, -122.4158], // 10: patient lingers at bench (dist ~85m)
      [37.7774, -122.4159], // 11: patient stays still (dist ~135m, warning tick 2)
      [37.7774, -122.4160], // 12: patient stays still (dist ~190m, tick 3)
      [37.7774, -122.4160], // 13: patient stays still (dist ~250m, tick 4 -> ALERT)
      [37.7774, -122.4161], // 14: patient still separated
      [37.7775, -122.4160], // 15: patient begins walking towards companions
      [37.7775, -122.4159], // 16: closing gap
      [37.7775, -122.4158], // 17: rejoined with companions (<50m threshold)
      [37.7769, -122.4165], // 18: walking together
      [37.7754, -122.4182], // 19: walking back together towards zone
    ],
  },
  sim_user_3: {
    id: 'sim_user_3',
    label: 'Simulated User 3',
    defaultRole: 'companion',
    badge: '3',
    color: '#d97706', // Warm Amber
    dashArray: '3, 6',
    waypoints: [
      [37.7721, -122.4226], // 0: inside zone
      [37.7726, -122.4219], // 1: inside zone
      [37.7732, -122.4211], // 2: inside zone
      [37.7739, -122.4203], // 3: near zone boundary
      [37.7745, -122.4195], // 4: crossing boundary together
      [37.7751, -122.4188], // 5: outside zone together
      [37.7757, -122.4181], // 6: outside zone together
      [37.7763, -122.4174], // 7: outside zone together
      [37.7769, -122.4167], // 8: outside zone together
      [37.7775, -122.4160], // 9: outside zone together
      [37.7782, -122.4152], // 10: companions advance
      [37.7788, -122.4144], // 11: companions advance
      [37.7794, -122.4137], // 12: companions advance
      [37.7799, -122.4131], // 13: companions advance
      [37.7802, -122.4126], // 14: companions halt/turn back
      [37.7794, -122.4136], // 15: returning to patient
      [37.7785, -122.4147], // 16: returning
      [37.7777, -122.4157], // 17: companions rejoin patient
      [37.7771, -122.4164], // 18: reunited
      [37.7756, -122.4181], // 19: walking back together towards zone
    ],
  },
}

export const TOTAL_DEMO_STEPS = DEMO_ROUTES.sim_user_1.waypoints.length

/**
 * Retrieve simulated users at a specific animation step.
 * Returns array with updated current coordinate and trail up to this step.
 */
export function getSimulatedUsersAtStep(stepIndex = 0, selectedPatientId = 'sim_user_2') {
  const boundedStep = Math.max(0, Math.min(stepIndex, TOTAL_DEMO_STEPS - 1))

  return Object.values(DEMO_ROUTES).map((route) => {
    const isPatient = route.id === selectedPatientId
    const currentCoord = route.waypoints[boundedStep]
    const trailSoFar = route.waypoints.slice(0, boundedStep + 1)

    return {
      id: route.id,
      label: route.label,
      badge: route.badge,
      color: route.color,
      dashArray: route.dashArray,
      isPatient,
      role: isPatient ? 'PATIENT' : 'COMPANION',
      currentCoord,
      trail: trailSoFar,
      fullTrail: route.waypoints,
    }
  })
}

/**
 * Group Cohesion and Separation State Evaluation.
 *
 * @param {Object} params
 * @param {Array} params.users - current simulated user list
 * @param {string} params.selectedPatientId - user ID designated as patient
 * @param {Object} params.geofence - geofence definition
 * @param {number} params.distanceThresholdMeters - separation threshold (e.g. 50m)
 * @param {number} params.sustainedWarningTicks - ticks of separation before Warning (e.g. 2)
 * @param {number} params.sustainedAlertTicks - ticks of separation before Alert (e.g. 4)
 * @param {number} params.currentSeparationTicks - ticks patient has been separated
 * @param {string|null} params.previousAlertLevel - 'NONE' | 'WARNING' | 'ALERT' | 'RESOLVED'
 */
export function evaluateGroupState({
  users,
  selectedPatientId = 'sim_user_2',
  geofence = DEFAULT_GEOFENCE,
  distanceThresholdMeters = 50,
  sustainedWarningTicks = 2,
  sustainedAlertTicks = 4,
  currentSeparationTicks = 0,
  previousAlertLevel = 'NONE',
}) {
  const patient = users.find((u) => u.id === selectedPatientId) || users[0]
  const companions = users.filter((u) => u.id !== patient.id)

  const patientCoord = patient ? patient.currentCoord : null

  // 1. Geofence evaluation
  const patientInsideZone = patientCoord ? isInsideGeofence(patientCoord, geofence) : true
  const companionsInsideZone = companions.map((c) => isInsideGeofence(c.currentCoord, geofence))
  const allInsideZone = patientInsideZone && companionsInsideZone.every(Boolean)
  const allOutsideZone = !patientInsideZone && companionsInsideZone.every((inZone) => !inZone)

  // 2. Cohesion distance evaluation
  const companionDistances = companions.map((c) => ({
    companionId: c.id,
    label: c.label,
    distanceM: patientCoord ? haversineMeters(patientCoord, c.currentCoord) : 0,
  }))

  const maxPatientCompanionDist = companionDistances.length > 0
    ? Math.max(...companionDistances.map((d) => d.distanceM))
    : 0

  // Check pairwise distances among all users to verify group cohesion
  const pairwiseDistances = []
  for (let i = 0; i < users.length; i++) {
    for (let j = i + 1; j < users.length; j++) {
      pairwiseDistances.push(haversineMeters(users[i].currentCoord, users[j].currentCoord))
    }
  }
  const maxPairwiseDist = pairwiseDistances.length > 0 ? Math.max(...pairwiseDistances) : 0

  // Group is cohesive if max distance from patient to any companion is within threshold
  const isCohesive = maxPatientCompanionDist <= distanceThresholdMeters
  const isPatientSeparated = !isCohesive

  // 3. Separation Duration and Escalation Logic
  let nextSeparationTicks = currentSeparationTicks
  let alertLevel = 'NONE' // 'NONE' | 'WARNING' | 'ALERT' | 'RESOLVED'

  if (isPatientSeparated) {
    nextSeparationTicks = currentSeparationTicks + 1
    if (nextSeparationTicks >= sustainedAlertTicks) {
      alertLevel = 'ALERT'
    } else if (nextSeparationTicks >= sustainedWarningTicks) {
      alertLevel = 'WARNING'
    } else {
      alertLevel = 'NONE' // brief separation under warning duration threshold
    }
  } else {
    // Patient has rejoined (dist <= threshold)
    if (previousAlertLevel === 'ALERT' || previousAlertLevel === 'WARNING' || currentSeparationTicks >= sustainedWarningTicks) {
      alertLevel = 'RESOLVED'
    } else {
      alertLevel = 'NONE'
    }
    nextSeparationTicks = 0
  }

  // 4. Zone Status Description
  let zoneStatusText = 'Inside Safe Zone'
  let zoneStatusType = 'safe' // 'safe' | 'outside-together' | 'outside-separated'

  if (!patientInsideZone) {
    if (isCohesive) {
      zoneStatusText = 'Outside Safe Zone (Group Travelling Together)'
      zoneStatusType = 'outside-together'
    } else {
      zoneStatusText = 'Outside Safe Zone (Separated)'
      zoneStatusType = 'outside-separated'
    }
  }

  // 5. Cohesion & Group Status Message (Requirement 5, 6, 7)
  let statusMessage = ''
  let statusBadgeType = 'normal' // 'normal' | 'info' | 'warning' | 'alert' | 'resolved'

  if (alertLevel === 'ALERT') {
    statusMessage = `🚨 DEMO ALERT: Group Separation Alert — ${patient.label} separated by ${Math.round(maxPatientCompanionDist)}m (sustained ${nextSeparationTicks} ticks)`
    statusBadgeType = 'alert'
  } else if (alertLevel === 'WARNING') {
    statusMessage = `⚠️ DEMO WARNING: Group Separation Warning — ${patient.label} separated by ${Math.round(maxPatientCompanionDist)}m (sustained ${nextSeparationTicks} ticks)`
    statusBadgeType = 'warning'
  } else if (alertLevel === 'RESOLVED') {
    statusMessage = `✅ DEMO NOTICE: Alert Resolved — ${patient.label} has rejoined group (distance ${Math.round(maxPatientCompanionDist)}m)`
    statusBadgeType = 'resolved'
  } else if (!patientInsideZone && isCohesive) {
    // Requirement 5: Leaving geofence while together must NOT trigger emergency alert
    statusMessage = 'ℹ️ Group travelling together outside zone (Safe Zone departed with companions — no emergency alert)'
    statusBadgeType = 'info'
  } else if (allInsideZone && isCohesive) {
    statusMessage = 'Group travelling together inside safe zone (Cohesive)'
    statusBadgeType = 'normal'
  } else {
    statusMessage = `Group active · Separation: ${Math.round(maxPatientCompanionDist)}m`
    statusBadgeType = 'normal'
  }

  return {
    patient,
    selectedPatientId: patient.id,
    companions,
    companionDistances,
    maxDistanceM: maxPatientCompanionDist,
    maxPairwiseDist,
    distanceThresholdMeters,
    isCohesive,
    isPatientSeparated,
    patientInsideZone,
    allInsideZone,
    allOutsideZone,
    zoneStatusText,
    zoneStatusType,
    separationTicks: nextSeparationTicks,
    alertLevel,
    statusMessage,
    statusBadgeType,
  }
}
