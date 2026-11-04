"""Today's forecast for the morning brief (phase 11), from Open-Meteo (no API key; api.open-meteo.com allowlisted)."""
from typing import Any

import httpx

URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (Open-Meteo `weather_code`)
WMO = {0: "晴", 1: "晴時多雲", 2: "多雲", 3: "陰", 45: "霧", 48: "霧", 51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
       56: "凍毛毛雨", 57: "凍毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 66: "凍雨", 67: "凍雨", 71: "小雪",
       73: "中雪", 75: "大雪", 77: "雪粒", 80: "陣雨", 81: "陣雨", 82: "強陣雨", 85: "陣雪", 86: "陣雪",
       95: "雷雨", 96: "雷雨伴冰雹", 99: "雷雨伴冰雹"}


def line(name: str, daily: dict[str, Any]) -> str:
    code = int(daily["weather_code"][0])
    lo, hi = round(daily["temperature_2m_min"][0]), round(daily["temperature_2m_max"][0])
    rain = daily.get("precipitation_probability_max", [None])[0]
    text = f"{name} {WMO.get(code, '—')} {lo}–{hi}°C"
    return text + (f" 降雨 {rain}%" if rain is not None else "") + (" ☂️" if (rain or 0) >= 50 else "")


async def today(places: list[dict[str, Any]], timeout_s: float = 10) -> list[str]:
    """One line per place; a place whose forecast fails says so instead of failing the brief."""
    out = []
    async with httpx.AsyncClient(timeout=timeout_s) as http:
        for p in places:
            try:
                resp = await http.get(URL, params={
                    "latitude": p["lat"], "longitude": p["lon"], "timezone": "Asia/Taipei", "forecast_days": 1,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})
                resp.raise_for_status()
                out.append(line(p["name"], resp.json()["daily"]))
            except Exception as e:  # noqa: BLE001
                out.append(f"{p['name']} 天氣暫時無法取得 ({type(e).__name__})")
    return out
