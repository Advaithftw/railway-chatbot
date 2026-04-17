const normalizeClass = (travelClass = "") => {
  const cls = String(travelClass).toLowerCase().replace(/\s+/g, "").trim();

  if (cls.includes("1ac") || cls === "a1") return "1AC";
  if (cls.includes("2ac") || cls === "a2") return "2AC";
  if (cls.includes("3ac") || cls === "a3") return "3AC";
  if (cls.includes("sl") || cls.includes("sleeper")) return "Sleeper";

  return "3AC";
};

export const buildFeatures = (entities = {}, kgData = {}, trainNumber = null, trainType = null) => {
  const waitlist = Number(entities.waitlist);
  const stops = Number(kgData.stops);
  const distance = Number(kgData.distance);
  const time = Number(kgData.time);
  const trainCount = Number(kgData.trainCount);

  const parsedWaitlist = Number.isFinite(waitlist) ? waitlist : 50;
  const parsedTrainCount = Number.isFinite(trainCount) ? trainCount : 0;
  const seatAvailability = Math.max(0, parsedTrainCount * 40 - parsedWaitlist);

  const features = {
    "Waitlist Position": parsedWaitlist,
    "Class of Travel": normalizeClass(entities.class),
    "Quota": "General",
    "Train Type": trainType || "Express",
    "Travel Distance": Number.isFinite(distance) ? distance : 800,
    "Number of Stations": Number.isFinite(stops) ? stops : 12,
    "Travel Time": Number.isFinite(time) ? time : 16,
    "Seat Availability": seatAvailability,
    "Holiday or Peak Season": 0,
    "Source Station": entities.source || "Unknown",
    "Destination Station": entities.destination || "Unknown"
  };

  if (trainNumber) {
    features["Train Number"] = Number(trainNumber);
    features.train_number = trainNumber;
  }

  if (trainType) {
    features.train_type = trainType;
  }

  return features;
};
