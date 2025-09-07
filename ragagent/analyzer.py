"""
レイヤ分類とフレームワーク検出のためのユーティリティクラス
"""

import re
from typing import List
from .models import LayerType, FrameworkType, SourceFile

class LayerClassifier:
    """レイヤ分類クラス"""
    def __init__(self):
        self.layer_patterns = {
            LayerType.VIEW: [
                r'\.html$', r'\.jsx$', r'\.vue$', r'\.tsx$',
                r'template', r'view', r'component',
                r'@Component', r'render\(', r'<template>'
            ],
            LayerType.CONTROLLER: [
                r'Controller', r'@RestController', r'@Controller',
                r'views\.py', r'@app\.route', r'router',
                r'express\.Router', r'def\s+\w+\(request'
            ],
            LayerType.SERVICE: [
                r'Service', r'@Service', r'Business',
                r'Logic', r'UseCase', r'Application'
            ],
            LayerType.REPOSITORY: [
                r'Repository', r'@Repository', r'DAO',
                r'models\.py', r'Model', r'Entity',
                r'@Entity', r'ActiveRecord'
            ],
            LayerType.DOMAIN: [
                r'Domain', r'Entity', r'ValueObject',
                r'Aggregate', r'DomainService'
            ],
            LayerType.CONFIG: [
                r'config', r'Config', r'settings',
                r'application\.properties', r'\.env'
            ],
            LayerType.UTIL: [
                r'util', r'Util', r'helper', r'Helper',
                r'common', r'Common'
            ]
        }

    def classify_layer(self, file: SourceFile) -> LayerType:
        """ファイルのレイヤを分類"""
        content_and_path = f"{file.path} {file.content}"

        for layer, patterns in self.layer_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_and_path, re.IGNORECASE):
                    return layer

        return LayerType.UNKNOWN

class FrameworkDetector:
    """フレームワーク検出クラス"""

    def __init__(self):
        self.framework_patterns = {
            FrameworkType.SPRING_BOOT: [
                r'@SpringBootApplication',
                r'@RestController',
                r'@Service',
                r'@Repository',
                r'springframework'
            ],
            FrameworkType.DJANGO: [
                r'from django',
                r'django.db.models',
                r'django.views',
                r'django.urls'
            ],
            FrameworkType.FLASK: [
                r'from flask',
                r'Flask\(',
                r'@app.route'
            ],
            FrameworkType.REACT: [
                r'import React',
                r'from [\'"]react[\'"]',
                r'jsx',
                r'useState',
                r'useEffect'
            ],
            FrameworkType.VUE: [
                r'import Vue',
                r'from [\'"]vue[\'"]',
                r'\.vue$'
            ],
            FrameworkType.EXPRESS: [
                r'express\(',
                r'app\.get\(',
                r'app\.post\('
            ],
            FrameworkType.LARAVEL: [
                r'use Illuminate',
                r'extends Controller',
                r'Eloquent'
            ],
            FrameworkType.RAILS: [
                r'ActionController::Base',
                r'ActiveRecord::Base',
                r'Rails\.application'
            ]
        }

    def detect_framework(self, files: List[SourceFile]) -> FrameworkType:
        """ファイル群からフレームワークを検出"""
        framework_scores = {fw: 0 for fw in FrameworkType}

        for file in files:
            for framework, patterns in self.framework_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, file.content, re.IGNORECASE):
                        framework_scores[framework] += 1

        max_score = max(framework_scores.values())
        if max_score == 0:
            return FrameworkType.UNKNOWN

        return max(framework_scores, key=lambda fw: framework_scores[fw])
