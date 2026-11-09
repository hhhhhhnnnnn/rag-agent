"""生成合成规模语料：星尘智能家居生态产品目录（种子固定，可复现）。

产出:
  data/星尘智能家居-<类别>.md ×6   —— 入检索库的产品目录
  eval/scale_truth.json            —— 出题用答案键（每款产品全部属性，不入库）

设计约束（为什么这么造）:
  - 每款产品一块（~350字），事实分散、无跨产品共置——聚合题必须多块才能答
  - 价格全局唯一（排序题无平局）、NFC 只在部分类别出现（过滤集合小而确定）
  - 绝不生成任何"排行榜/汇总表"文档——那会让单发检索直接答掉聚合题
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
random.seed(42)  # 固定种子：任何人重跑得到完全相同的语料

CATS = {
    "智能音箱": {
        "series": ["韵声", "聆语", "回响"],
        "price": (199, 899),
        "weight": (250, 900),
        "extra": ("拾音距离", ["3 米", "5 米", "7 米"]),
        "nfc_pool": [True, True, False, False, False],  # 音箱系 40% 支持 NFC
        "faqs": [
            "唤醒不灵敏时，请在 App 中重新训练声纹",
            "夜间可开启勿扰模式，呼吸灯熄灭属正常",
            "多房间联动需在 App 中创建音箱编组",
        ],
    },
    "智能灯具": {
        "series": ["晨曦", "暮光", "星轨"],
        "price": (79, 499),
        "weight": (150, 1200),
        "extra": ("色温范围", ["2700K-6500K", "3000K-6000K", "1800K-6500K"]),
        "nfc_pool": [False],
        "faqs": [
            "频闪请检查调光器兼容性列表",
            "分组控制需先将灯具加入同一房间",
            "夜灯模式可在断网时本地触发",
        ],
    },
    "智能插座": {
        "series": ["恒流", "安培"],
        "price": (39, 199),
        "weight": (60, 200),
        "extra": ("额定功率", ["10A/2200W", "16A/3500W"]),
        "nfc_pool": [False],
        "faqs": [
            "离线请长按侧面按键 5 秒重置网络",
            "计量误差约 ±1%，不作电费结算依据",
            "定时任务在断网后仍本地执行",
        ],
    },
    "智能传感器": {
        "series": ["哨兵", "微尘"],
        "price": (59, 299),
        "weight": (30, 150),
        "extra": ("电池续航", ["6 个月", "12 个月", "24 个月"]),
        "nfc_pool": [False],
        "faqs": [
            "误报频繁请调整灵敏度档位",
            "低温环境电池续航会缩短约三成",
            "配对失败请更换电池后重试",
        ],
    },
    "智能安防": {
        "series": ["守望", "夜枭"],
        "price": (199, 799),
        "weight": (200, 800),
        "extra": ("夜视距离", ["5 米", "8 米", "10 米"]),
        "nfc_pool": [True, False, False, False],  # 安防系 25% 支持 NFC
        "faqs": [
            "画面卡顿请切换主码流为子码流",
            "存储卡满后自动循环覆盖最旧录像",
            "移动侦测区域可在 App 中框选",
        ],
    },
    "智能影音": {
        "series": ["幕光", "声场"],
        "price": (299, 1499),
        "weight": (300, 1500),
        "extra": ("解码格式", ["Dolby Audio", "DTS-HD", "双声道 LPCM"]),
        "nfc_pool": [True, False, False],  # 影音系 33% 支持 NFC
        "faqs": [
            "HDMI 无输出请检查 EDID 透传设置",
            "回音壁建议距墙 10cm 以上摆放",
            "固件升级期间请勿断电",
        ],
    },
}

CONN = ["Wi-Fi 双频", "Wi-Fi 2.4GHz", "蓝牙 Mesh", "Zigbee 3.0", "Wi-Fi+蓝牙双模"]
WARRANTY = [12, 12, 12, 18, 18, 24]  # 50% 12月 / 33% 18月 / 17% 24月
PER_CAT = 15
SELLING = [
    "同档位性价比之选", "为小户型优化", "主打长续航", "主打安装免工具",
    "金属机身更耐用", "低功耗芯片方案", "年度迭代款", "入门尝鲜首选",
]


def main() -> None:
    used_names, used_prices = set(), set()
    truth = []

    def uniq_price(lo: int, hi: int) -> int:
        while True:
            p = random.randint(lo, hi)
            if p not in used_prices:
                used_prices.add(p)
                return p

    for cat, cfg in CATS.items():
        lines = [f"# 星尘智能家居产品目录 · {cat}\n"]
        made = 0
        while made < PER_CAT:
            series = cfg["series"][made % len(cfg["series"])]
            model = f"{chr(65 + made // len(cfg['series']))}{made % len(cfg['series']) + 1}"
            name = f"星尘 {series} {model}"
            if name in used_names:
                continue
            used_names.add(name)
            price = uniq_price(*cfg["price"])
            weight = random.randint(*cfg["weight"])
            conn = random.choice(CONN)
            warranty = random.choice(WARRANTY)
            nfc = random.choice(cfg["nfc_pool"])
            extra_val = random.choice(cfg["extra"][1])
            selling = random.choice(SELLING)
            faq = random.choice(cfg["faqs"])
            lines += [
                f"### {name}\n",
                f"- 售价：{price} 元",
                f"- 重量：{weight} g",
                f"- 连接方式：{conn}",
                f"- 保修：整机 {warranty} 个月",
                f"- {cfg['extra'][0]}：{extra_val}",
                f"- NFC 一碰配对：{'支持' if nfc else '不支持'}",
                f"- 卖点：{selling}",
                f"- 常见问题：{faq}\n",
            ]
            truth.append(dict(name=name, category=cat, price=price, weight=weight,
                              conn=conn, warranty=warranty, nfc=nfc,
                              extra_label=cfg["extra"][0], extra=extra_val))
            made += 1
        out = ROOT / "data" / f"星尘智能家居-{cat}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"[gen] {out.name}: {PER_CAT} 款")

    # ---- 自检（校验不过直接抛，不产出半成品答案键） ----
    assert len(truth) == PER_CAT * len(CATS), "产品总数不对"
    assert len({t['name'] for t in truth}) == len(truth), "型号重名"
    assert len({t['price'] for t in truth}) == len(truth), "价格有平局"
    nfc_cnt = sum(t['nfc'] for t in truth)
    assert 5 <= nfc_cnt <= 20, "NFC 集合过大/过小，聚合题不可控"

    (ROOT / "eval" / "scale_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[gen] 共 {len(truth)} 款 | NFC 支持 {nfc_cnt} 款 | 答案键 -> eval/scale_truth.json")


if __name__ == "__main__":
    main()
