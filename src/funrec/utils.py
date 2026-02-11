# -*- coding:utf-8 -*-
""" """

import json
from threading import Thread

import requests
from funlog import getLogger
from packaging.version import parse
from pip._vendor.packaging.version import parse

logger = getLogger("funrec")


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = "https://pypi.python.org/pypi/deepctr-torch/json"
            req = requests.get(url_pattern)
            latest_version = parse("0")
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode("utf-8"))
                releases = j.get("releases", [])
                for release in releases:
                    ver = parse(release)
                    if ver.is_prerelease or ver.is_postrelease:
                        continue
                    latest_version = max(latest_version, ver)
                if latest_version > version:
                    logger.warning(
                        "\nDeepCTR-PyTorch version {0} detected. Your version is {1}.\nUse `pip install -U deepctr-torch` to upgrade.Changelog: https://github.com/shenweichen/DeepCTR-Torch/releases/tag/v{0}".format(
                            latest_version, version
                        )
                    )
        except Exception as e:
            logger.error(e)
            logger.error(
                "Please check the latest version manually on https://pypi.org/project/deepctr-torch/#history"
            )
            return

    Thread(target=check, args=(version,)).start()
