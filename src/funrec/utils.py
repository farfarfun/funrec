"""funrec 通用工具。"""

import json
from threading import Thread

import requests
from farlog import getLogger
from packaging.version import InvalidVersion, parse

logger = getLogger("funrec")


def check_version(version: str) -> None:
    """后台检查 PyPI 上的最新版本，不阻塞调用方。"""

    def check(current_version: str) -> None:
        url = "https://pypi.org/pypi/funrec/json"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "检查 funrec 版本失败，URL={}，当前版本={}：{}",
                url,
                current_version,
                exc,
            )
            return

        try:
            releases = json.loads(response.text).get("releases", {})
            installed_version = parse(current_version)
        except (json.JSONDecodeError, AttributeError, TypeError, InvalidVersion) as exc:
            logger.error(
                "解析 funrec 版本响应失败，URL={}，当前版本={}：{}",
                url,
                current_version,
                exc,
            )
            return

        latest_version = parse("0")
        for release in releases:
            try:
                candidate = parse(release)
            except (InvalidVersion, TypeError):
                logger.warning("跳过无法解析的 funrec 版本：{}", release)
                continue
            if not candidate.is_prerelease and not candidate.is_postrelease:
                latest_version = max(latest_version, candidate)

        if latest_version > installed_version:
            logger.warning(
                "检测到 funrec 新版本 {}，当前版本 {}。请运行 `pip install -U funrec` 升级。",
                latest_version,
                installed_version,
            )

    Thread(target=check, args=(version,), daemon=True).start()
