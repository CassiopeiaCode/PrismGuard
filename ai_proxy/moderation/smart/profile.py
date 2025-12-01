"""
智能审核配置文件管理
"""
import json
import os
from typing import Optional
from pydantic import BaseModel


class AIConfig(BaseModel):
    """AI 配置"""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key_env: str = "MOD_AI_API_KEY"
    timeout: int = 10


class PromptConfig(BaseModel):
    """提示词配置"""
    template_file: str = "ai_prompt.txt"
    max_text_length: int = 4000


class ProbabilityConfig(BaseModel):
    """概率配置"""
    ai_review_rate: float = 0.3
    random_seed: int = 42
    # 新增：阈值配置
    low_risk_threshold: float = 0.2   # p < 0.2 直接判安全
    high_risk_threshold: float = 0.8  # p > 0.8 直接判违规


class BoWTrainingConfig(BaseModel):
    """词袋模型训练配置"""
    min_samples: int = 200
    retrain_interval_minutes: int = 60
    max_samples: int = 50000
    
    # 特征配置
    max_features: int = 8000
    use_char_ngram: bool = True
    char_ngram_range: list = [2, 3]
    use_word_ngram: bool = True
    word_ngram_range: list = [1, 2]
    
    # 模型类型
    model_type: str = "sgd_logistic"  # sgd_logistic / logistic


class ProfileConfig(BaseModel):
    """配置文件结构"""
    ai: AIConfig = AIConfig()
    prompt: PromptConfig = PromptConfig()
    probability: ProbabilityConfig = ProbabilityConfig()
    bow_training: BoWTrainingConfig = BoWTrainingConfig()


class ModerationProfile:
    """审核配置文件"""
    
    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.base_dir = f"configs/mod_profiles/{profile_name}"
        self.config = self._load_config()
    
    def _load_config(self) -> ProfileConfig:
        """加载配置文件"""
        config_path = os.path.join(self.base_dir, "profile.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
                return ProfileConfig(**data)
        return ProfileConfig()
    
    def get_prompt_template(self) -> str:
        """获取提示词模板"""
        template_path = os.path.join(self.base_dir, self.config.prompt.template_file)
        if os.path.exists(template_path):
            with open(template_path, encoding="utf-8") as f:
                return f.read()
        return "请判断以下文本是否违规：\n{{text}}"
    
    def truncate_text(self, text: str) -> str:
        """根据配置截断文本"""
        max_len = self.config.prompt.max_text_length
        if len(text) > max_len:
            front_len = int(max_len * 2 / 3)
            back_len = int(max_len * 1 / 3)
            front_part = text[:front_len]
            back_part = text[-back_len:]
            text = front_part + "\n...[中间省略]...\n" + back_part
            print(f"[DEBUG] 文本截断: 原长度={len(text)}, 保留前{front_len}+后{back_len}字符")
        return text
    
    def render_prompt(self, text: str) -> str:
        """渲染提示词（带 HTML 转义）"""
        import html
        
        template = self.get_prompt_template()
        
        # HTML 转义特殊字符，防止注入攻击
        escaped_text = html.escape(text)
        
        return template.replace("{{text}}", escaped_text)
    
    def get_db_path(self) -> str:
        """获取数据库路径"""
        return os.path.join(self.base_dir, "history.db")
    
    def get_model_path(self) -> str:
        """获取模型路径"""
        return os.path.join(self.base_dir, "bow_model.pkl")
    
    def get_vectorizer_path(self) -> str:
        """获取向量化器路径"""
        return os.path.join(self.base_dir, "bow_vectorizer.pkl")
    
    def bow_model_exists(self) -> bool:
        """检查词袋模型是否存在"""
        return (os.path.exists(self.get_model_path()) and 
                os.path.exists(self.get_vectorizer_path()))


# 全局配置缓存
_profiles = {}


def get_profile(profile_name: str) -> ModerationProfile:
    """获取或创建配置"""
    if profile_name not in _profiles:
        _profiles[profile_name] = ModerationProfile(profile_name)
    return _profiles[profile_name]