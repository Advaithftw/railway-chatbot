export const extractFeatures = (query) => {
  const wlMatch = query.match(/\d+/);

  return {
    "Waitlist Position": wlMatch ? parseInt(wlMatch[0]) : 50,
    "Class of Travel": query.includes("sleeper") ? "Sleeper" : "3AC",
    "Quota": "General",
    "Train Type": "Express",
    "Travel Distance": 1200, // placeholder
    "Number of Stations": 15, // placeholder
    "Travel Time": 24, // placeholder
    "Seat Availability": 200,
    "Holiday or Peak Season": 0,
    "Source Station": "Delhi",
    "Destination Station": "Mumbai"
  };
};