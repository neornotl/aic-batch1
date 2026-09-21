"""Query expansion and language variants without requiring an LLM."""
from __future__ import annotations
import re

VI_EN = {"người": "person people", "xe": "car vehicle", "đỏ": "red", "xanh": "blue",
         "đứng": "standing", "ngồi": "sitting", "đi": "walking", "nói": "speaking talk",
         "nhà": "house building", "biển": "sea sign", "màn hình": "screen display",
         "tàu vũ trụ": "spacecraft spaceship", "phi hành gia": "astronaut", "áo đen": "black shirt",
         "cực quang": "aurora", "lễ hội": "festival", "ẩm thực": "food cuisine",
         "bạch tuộc": "octopus", "con mực": "squid", "cô bé": "girl", "túi giấy": "paper bag",
         "phóng": "launch", "nhiệm vụ": "mission", "vùng cực": "polar",
         # Camera language and temporal cues commonly used by BTC.
         "bắt đầu": "beginning starts", "kết thúc": "ending ends", "cảnh quay chậm": "slow motion",
         "từ trên cao": "aerial drone overhead", "toàn cảnh": "wide shot", "cận cảnh": "close-up",
         "góc máy sát mặt đường": "low angle ground level", "ghi hình": "filming recording",
         "máy quay": "camera filming", "rọi đèn": "shines light flashlight", "dưới nước": "underwater",
         "bình minh": "dawn sunrise", "ban đêm": "night nighttime", "trời mưa": "rain rainy",
         # Actions and settings.
         "kéo lưới": "pulling fishing net", "lưới cá": "fishing net", "đánh cá": "fishing",
         "lội nước": "flooded wading", "qua cầu": "crossing bridge", "biển báo": "road sign",
         "trạm xăng": "gas station fuel station", "xe ôm công nghệ": "ride hailing motorcycle driver",
         "bình xăng": "fuel tank", "giá dầu": "fuel price", "vạch đích": "finish line",
         "xe đạp": "bicycle cycling", "tay đua": "cyclist racer", "về đích": "finishing race",
         "xếp thành hàng": "standing in line", "tập thể dục": "exercise workout",
         "chạm mũi chân": "touching toes", "đeo kính": "wearing glasses", "đội nón": "wearing hat",
         # Food preparation.
         "đậu hà lan": "peas", "hành tây": "onion", "ớt đỏ": "red chili",
         "bếp lửa": "flame stove", "chảo": "frying pan", "cà rốt": "carrot",
         "rau củ": "vegetables", "đậu bắp": "okra", "súp lơ": "broccoli",
         "đũa": "chopsticks", "hấp": "steamed", "cắt nho": "cutting grapes",
         "chùm nho": "bunch grapes", "giàn nho": "grapevine", "cái kéo": "scissors",
         # Subjects that recur in the public AIC corpus.
         "sư tử": "lion lions", "sở thú": "zoo", "bảng thông tin": "information sign",
         "nhân viên": "staff worker", "áo xanh lá": "green shirt", "cân": "scale weighing",
         "bản đồ": "map", "công trình thủy lợi": "waterworks irrigation", "con đập": "dam",
         "đá quý": "gemstone gem", "mỏ đá": "open pit mine quarry", "khăn trùm đầu": "headscarf",
         "cá": "fish", "cầu": "bridge", "động đất": "earthquake", "bảng chú giải": "map legend",
         "tâm chấn": "epicenter", "cấp độ": "magnitude", "vị trí": "location point"}

def expand_query(query: str, remote: list[str] | None = None) -> list[str]:
    q = " ".join(query.strip().split())
    variants = [q]
    translated = q.lower()
    for vi, en in sorted(VI_EN.items(), key=lambda x: -len(x[0])):
        translated = translated.replace(vi, en)
    if translated != q.lower(): variants.append(translated)
    tokens = re.findall(r"[\wÀ-ỹ]+", q.lower())
    if len(tokens) > 1: variants.append(" ".join(tokens))
    for value in remote or []:
        if value and value not in variants: variants.append(value)
    return variants


def clip_english_query(query: str) -> str:
    """Build a pure-English keyword query for CLIP ViT-B/32."""
    low = query.lower()
    parts: list[str] = []
    for vi, en in sorted(VI_EN.items(), key=lambda x: -len(x[0])):
        if vi in low:
            parts.append(en)
    # Fallback: if no Vietnamese keywords matched, use the translated variant
    if not parts:
        variants = expand_query(query)
        if len(variants) > 1:
            # translated variant may still contain Vietnamese; keep only ascii
            return " ".join(word for word in variants[1].split() if word.isascii())
        return query
    return " ".join(parts)
