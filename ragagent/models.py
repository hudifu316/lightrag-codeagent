"""
RAGエージェントのモデル定義
アーキテクチャレイヤ、フレームワーク、ソースファイル情報のデータモデル
"""

from enum import Enum
from dataclasses import dataclass
from typing import Set

class LayerType(Enum):
    """アーキテクチャレイヤの種類"""
    VIEW = "view"
    CONTROLLER = "controller"
    SERVICE = "service"
    REPOSITORY = "repository"
    DOMAIN = "domain"
    UTIL = "util"
    CONFIG = "config"
    UNKNOWN = "unknown"

class FrameworkType(Enum):
    """フレームワークの種類"""
    SPRING_BOOT = "spring_boot"
    DJANGO = "django"
    FLASK = "flask"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    EXPRESS = "express"
    LARAVEL = "laravel"
    RAILS = "rails"
    UNKNOWN = "unknown"

@dataclass
class SourceFile:
    """ソースファイル情報"""
    path: str
    content: str
    extension: str
    layer: LayerType
    dependencies: Set[str]
    framework: FrameworkType
