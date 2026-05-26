# Sisyphean skill — geocoding, routing, and nearby POI search via OpenStreetMap/Overpass
import sys
import json
import math
import urllib.request
import urllib.parse


NOMINATIM = "https://nominatim.openstreetmap.org"
OSRM      = "http://router.project-osrm.org/route/v1/driving"
OVERPASS  = "https://overpass-api.de/api/interpreter"
HEADERS   = {"User-Agent": "Sisyphean/1.0"}


def fetch(url: str, *, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def geocode(query: str) -> tuple[float, float, str] | None:
    url = f"{NOMINATIM}/search?q={urllib.parse.quote(query)}&format=json&limit=1"
    raw = fetch(url)
    data = json.loads(raw)
    if not data:
        return None
    r = data[0]
    return float(r["lat"]), float(r["lon"]), r.get("display_name", "")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def cmd_geocode(query: str) -> None:
    try:
        result = geocode(query)
    except Exception as e:
        print(f"Error: {e}")
        return
    if not result:
        print(f"No results for: {query}")
        return
    lat, lon, name = result
    print(f"Lat : {lat}")
    print(f"Lon : {lon}")
    print(f"Name: {name}")


def cmd_route(origin: str, destination: str) -> None:
    try:
        g1 = geocode(origin)
        g2 = geocode(destination)
    except Exception as e:
        print(f"Geocoding error: {e}")
        return
    if not g1:
        print(f"Could not geocode: {origin}")
        return
    if not g2:
        print(f"Could not geocode: {destination}")
        return
    lat1, lon1, name1 = g1
    lat2, lon2, name2 = g2
    print(f"From: {name1[:80]}")
    print(f"To  : {name2[:80]}")
    print()

    # Try OSRM for driving distance
    try:
        url = f"{OSRM}/{lon1},{lat1};{lon2},{lat2}?overview=false"
        raw = fetch(url, timeout=10)
        data = json.loads(raw)
        if data.get("code") == "Ok":
            route = data["routes"][0]
            dist_km = route["distance"] / 1000
            dur_h   = route["duration"] / 3600
            print(f"Distance (driving) : {dist_km:.1f} km")
            print(f"Duration (estimate): {dur_h:.1f} h")
            return
    except Exception:
        pass

    # Fallback: straight-line
    dist = haversine_km(lat1, lon1, lat2, lon2)
    # Rough driving estimate: straight-line * 1.3, avg 60 km/h
    drive_est = dist * 1.3
    dur_h = drive_est / 60
    print(f"Straight-line distance: {dist:.1f} km")
    print(f"Estimated driving     : ~{drive_est:.1f} km  (~{dur_h:.1f} h at 60 km/h)")
    print("(OSRM unavailable — straight-line estimate only)")


def cmd_nearby(keyword: str, latlon: str) -> None:
    try:
        lat_s, lon_s = latlon.split(",")
        lat, lon = float(lat_s.strip()), float(lon_s.strip())
    except ValueError:
        print(f"Invalid lat,lon: {latlon}  — expected format '28.6,77.2'")
        return

    radius = 1000  # metres
    query = f"""
[out:json][timeout:15];
node[name~"{keyword}",i](around:{radius},{lat},{lon});
out 10;
"""
    try:
        data_bytes = urllib.request.urlopen(
            urllib.request.Request(
                OVERPASS,
                data=urllib.parse.urlencode({"data": query}).encode(),
                headers=HEADERS,
                method="POST",
            ),
            timeout=20,
        ).read()
        data = json.loads(data_bytes)
    except Exception as e:
        print(f"Overpass error: {e}")
        return

    elements = data.get("elements", [])
    if not elements:
        print(f"No '{keyword}' found within {radius}m of {lat},{lon}")
        return

    print(f"Nearby '{keyword}' within {radius}m of {lat},{lon}:\n")
    for el in elements[:10]:
        tags = el.get("tags", {})
        name = tags.get("name", "(no name)")
        addr = tags.get("addr:full") or tags.get("addr:street", "")
        el_lat = el.get("lat", "")
        el_lon = el.get("lon", "")
        print(f"  {name}")
        if addr:
            print(f"    Address: {addr}")
        if el_lat and el_lon:
            d = haversine_km(lat, lon, float(el_lat), float(el_lon)) * 1000
            print(f"    Distance: ~{d:.0f} m")
        print()


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print('  python skills/maps.py geocode "Eiffel Tower Paris"')
        print('  python skills/maps.py route "New Delhi" "Mumbai"')
        print('  python skills/maps.py nearby "coffee" "28.6,77.2"')
        return

    cmd = sys.argv[1].lower()

    if cmd == "geocode":
        cmd_geocode(" ".join(sys.argv[2:]))

    elif cmd == "route":
        # Split remaining args on " to " or take first two quoted args
        rest = sys.argv[2:]
        if len(rest) >= 2:
            cmd_route(rest[0], rest[1])
        else:
            print('Usage: python skills/maps.py route "ORIGIN" "DESTINATION"')

    elif cmd == "nearby":
        if len(sys.argv) < 4:
            print('Usage: python skills/maps.py nearby "KEYWORD" "LAT,LON"')
            return
        cmd_nearby(sys.argv[2], sys.argv[3])

    else:
        print(f"Unknown command: {cmd}")
        print("Available: geocode, route, nearby")


if __name__ == "__main__":
    main()
