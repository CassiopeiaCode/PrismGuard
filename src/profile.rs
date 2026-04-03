use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_api_key_env")]
    pub api_key_env: String,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

impl AiConfig {
    pub fn get_model_candidates(&self) -> Vec<String> {
        self.model
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptConfig {
    #[serde(default = "default_template_file")]
    pub template_file: String,
    #[serde(default = "default_prompt_max_text_length")]
    pub max_text_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProbabilityConfig {
    #[serde(default = "default_ai_review_rate")]
    pub ai_review_rate: f64,
    #[serde(default = "default_random_seed")]
    pub random_seed: u64,
    #[serde(default = "default_low_risk_threshold")]
    pub low_risk_threshold: f64,
    #[serde(default = "default_high_risk_threshold")]
    pub high_risk_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VocabBucket {
    #[serde(default = "default_bucket_name")]
    pub name: String,
    #[serde(default)]
    pub min_doc_ratio: f64,
    #[serde(default = "default_one")]
    pub max_doc_ratio: f64,
    #[serde(default = "default_bucket_limit")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BowTrainingConfig {
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    #[serde(default = "default_retrain_interval_minutes")]
    pub retrain_interval_minutes: usize,
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
    #[serde(default = "default_sample_loading")]
    pub sample_loading: String,
    #[serde(default = "default_max_features")]
    pub max_features: usize,
    #[serde(default = "default_true")]
    pub use_char_ngram: bool,
    #[serde(default = "default_char_ngram_range")]
    pub char_ngram_range: Vec<usize>,
    #[serde(default = "default_true")]
    pub use_word_ngram: bool,
    #[serde(default = "default_word_ngram_range")]
    pub word_ngram_range: Vec<usize>,
    #[serde(default = "default_bow_model_type")]
    pub model_type: String,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_seconds")]
    pub max_seconds: usize,
    #[serde(default = "default_max_db_items")]
    pub max_db_items: usize,
    #[serde(default = "default_true")]
    pub use_layered_vocab: bool,
    #[serde(default = "default_vocab_buckets")]
    pub vocab_buckets: Vec<VocabBucket>,
}

impl Default for BowTrainingConfig {
    fn default() -> Self {
        Self {
            min_samples: default_min_samples(),
            retrain_interval_minutes: default_retrain_interval_minutes(),
            max_samples: default_max_samples(),
            sample_loading: default_sample_loading(),
            max_features: default_max_features(),
            use_char_ngram: true,
            char_ngram_range: default_char_ngram_range(),
            use_word_ngram: true,
            word_ngram_range: default_word_ngram_range(),
            model_type: default_bow_model_type(),
            batch_size: default_batch_size(),
            max_seconds: default_max_seconds(),
            max_db_items: default_max_db_items(),
            use_layered_vocab: true,
            vocab_buckets: default_vocab_buckets(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FastTextTrainingConfig {
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    #[serde(default = "default_retrain_interval_minutes")]
    pub retrain_interval_minutes: usize,
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
    #[serde(default = "default_max_db_items")]
    pub max_db_items: usize,
    #[serde(default = "default_sample_loading")]
    pub sample_loading: String,
    #[serde(default)]
    pub use_jieba: bool,
    #[serde(default)]
    pub use_tiktoken: bool,
    #[serde(default = "default_tiktoken_model")]
    pub tiktoken_model: String,
    #[serde(default = "default_fasttext_dim")]
    pub dim: usize,
    #[serde(default = "default_fasttext_lr")]
    pub lr: f64,
    #[serde(default = "default_fasttext_epoch")]
    pub epoch: usize,
    #[serde(default = "default_fasttext_word_ngrams")]
    pub word_ngrams: usize,
    #[serde(default = "default_fasttext_minn")]
    pub minn: usize,
    #[serde(default = "default_fasttext_maxn")]
    pub maxn: usize,
    #[serde(default = "default_fasttext_bucket")]
    pub bucket: usize,
    #[serde(default = "default_true")]
    pub quantize: bool,
    #[serde(default)]
    pub qnorm: bool,
    #[serde(default)]
    pub cutoff: usize,
    #[serde(default = "default_true")]
    pub retrain: bool,
}

impl Default for FastTextTrainingConfig {
    fn default() -> Self {
        Self {
            min_samples: default_min_samples(),
            retrain_interval_minutes: default_retrain_interval_minutes(),
            max_samples: default_max_samples(),
            max_db_items: default_max_db_items(),
            sample_loading: default_sample_loading(),
            use_jieba: false,
            use_tiktoken: false,
            tiktoken_model: default_tiktoken_model(),
            dim: default_fasttext_dim(),
            lr: default_fasttext_lr(),
            epoch: default_fasttext_epoch(),
            word_ngrams: default_fasttext_word_ngrams(),
            minn: default_fasttext_minn(),
            maxn: default_fasttext_maxn(),
            bucket: default_fasttext_bucket(),
            quantize: true,
            qnorm: false,
            cutoff: 0,
            retrain: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashLinearTrainingConfig {
    #[serde(default = "default_min_samples")]
    pub min_samples: usize,
    #[serde(default = "default_retrain_interval_minutes")]
    pub retrain_interval_minutes: usize,
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
    #[serde(default = "default_max_db_items")]
    pub max_db_items: usize,
    #[serde(default = "default_sample_loading")]
    pub sample_loading: String,
    #[serde(default = "default_hashlinear_analyzer")]
    pub analyzer: String,
    #[serde(default = "default_hashlinear_ngram_range")]
    pub ngram_range: Vec<usize>,
    #[serde(default = "default_hashlinear_n_features")]
    pub n_features: usize,
    #[serde(default)]
    pub alternate_sign: bool,
    #[serde(default = "default_hashlinear_norm")]
    pub norm: Option<String>,
    #[serde(default = "default_hashlinear_alpha")]
    pub alpha: f64,
    #[serde(default = "default_hashlinear_epochs")]
    pub epochs: usize,
    #[serde(default = "default_hashlinear_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_random_seed")]
    pub random_seed: u64,
    #[serde(default = "default_max_seconds")]
    pub max_seconds: usize,
    #[serde(default)]
    pub use_jieba: bool,
}

impl Default for HashLinearTrainingConfig {
    fn default() -> Self {
        Self {
            min_samples: default_min_samples(),
            retrain_interval_minutes: default_retrain_interval_minutes(),
            max_samples: default_max_samples(),
            max_db_items: default_max_db_items(),
            sample_loading: default_sample_loading(),
            analyzer: default_hashlinear_analyzer(),
            ngram_range: default_hashlinear_ngram_range(),
            n_features: default_hashlinear_n_features(),
            alternate_sign: false,
            norm: default_hashlinear_norm(),
            alpha: default_hashlinear_alpha(),
            epochs: default_hashlinear_epochs(),
            batch_size: default_hashlinear_batch_size(),
            random_seed: default_random_seed(),
            max_seconds: default_max_seconds(),
            use_jieba: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    #[serde(default)]
    pub ai: AiConfig,
    #[serde(default)]
    pub prompt: PromptConfig,
    #[serde(default)]
    pub probability: ProbabilityConfig,
    #[serde(default = "default_local_model_type")]
    pub local_model_type: String,
    #[serde(default)]
    pub bow_training: BowTrainingConfig,
    #[serde(default)]
    pub fasttext_training: FastTextTrainingConfig,
    #[serde(default)]
    pub hashlinear_training: HashLinearTrainingConfig,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            ai: AiConfig::default(),
            prompt: PromptConfig::default(),
            probability: ProbabilityConfig::default(),
            local_model_type: default_local_model_type(),
            bow_training: BowTrainingConfig::default(),
            fasttext_training: FastTextTrainingConfig::default(),
            hashlinear_training: HashLinearTrainingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModerationProfile {
    pub profile_name: String,
    pub base_dir: PathBuf,
    pub config: ProfileConfig,
}

impl ModerationProfile {
    pub fn load(root_dir: impl AsRef<Path>, profile_name: &str) -> Result<Self> {
        let base_dir = root_dir
            .as_ref()
            .join("configs")
            .join("mod_profiles")
            .join(profile_name);
        let config_path = base_dir.join("profile.json");
        let config = if config_path.exists() {
            let raw = fs::read_to_string(&config_path)
                .with_context(|| format!("failed to read {}", config_path.display()))?;
            serde_json::from_str(&raw)
                .with_context(|| format!("failed to parse {}", config_path.display()))?
        } else {
            ProfileConfig::default()
        };

        Ok(Self {
            profile_name: profile_name.to_string(),
            base_dir,
            config,
        })
    }

    pub fn history_rocks_path(&self) -> PathBuf {
        self.base_dir.join("history.rocks")
    }

    pub fn training_status_path(&self) -> PathBuf {
        self.base_dir.join(".train_status.json")
    }

    pub fn hashlinear_model_path(&self) -> PathBuf {
        self.base_dir.join("hashlinear_model.pkl")
    }

    pub fn bow_model_path(&self) -> PathBuf {
        self.base_dir.join("bow_model.pkl")
    }

    pub fn bow_vectorizer_path(&self) -> PathBuf {
        self.base_dir.join("bow_vectorizer.pkl")
    }

    pub fn fasttext_model_path(&self) -> PathBuf {
        self.base_dir.join("fasttext_model.bin")
    }

    pub fn get_prompt_template(&self) -> String {
        let template_path = self.base_dir.join(&self.config.prompt.template_file);
        if let Ok(raw) = fs::read_to_string(&template_path) {
            return raw;
        }
        "请判断以下文本是否违规：\n{{text}}".to_string()
    }

    pub fn truncate_text(&self, text: &str) -> String {
        let max_len = self.config.prompt.max_text_length;
        if text.chars().count() <= max_len {
            return text.to_string();
        }

        let front_len = max_len.saturating_mul(2) / 3;
        let back_len = max_len.saturating_sub(front_len);
        let chars = text.chars().collect::<Vec<_>>();
        let front = chars.iter().take(front_len).collect::<String>();
        let back = chars
            .iter()
            .rev()
            .take(back_len)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        format!("{front}\n...[中间省略]...\n{back}")
    }

    pub fn render_prompt(&self, text: &str) -> String {
        self.get_prompt_template()
            .replace("{{text}}", &escape_html(text))
    }

    pub fn local_model_exists(&self) -> bool {
        match self.config.local_model_type.as_str() {
            "fasttext" => self.fasttext_model_path().exists(),
            "hashlinear" => {
                self.hashlinear_model_path().exists()
                    || (self.base_dir.join("hashlinear_runtime.json").exists()
                        && self.base_dir.join("hashlinear_runtime.coef.f32").exists())
            }
            _ => self.bow_model_path().exists() && self.bow_vectorizer_path().exists(),
        }
    }

    pub fn training_status(&self) -> Option<Value> {
        let path = self.training_status_path();
        let raw = fs::read_to_string(path).ok()?;
        serde_json::from_str(&raw).ok()
    }
}

fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn default_provider() -> String { "openai".to_string() }
fn default_base_url() -> String { "https://api.openai.com/v1".to_string() }
fn default_model() -> String { "gpt-4o-mini".to_string() }
fn default_api_key_env() -> String { "MOD_AI_API_KEY".to_string() }
fn default_timeout() -> u64 { 10 }
fn default_max_retries() -> u32 { 2 }
fn default_template_file() -> String { "ai_prompt.txt".to_string() }
fn default_prompt_max_text_length() -> usize { 4000 }
fn default_ai_review_rate() -> f64 { 0.3 }
fn default_random_seed() -> u64 { 42 }
fn default_low_risk_threshold() -> f64 { 0.2 }
fn default_high_risk_threshold() -> f64 { 0.8 }
fn default_min_samples() -> usize { 200 }
fn default_retrain_interval_minutes() -> usize { 60 }
fn default_max_samples() -> usize { 50_000 }
fn default_sample_loading() -> String { "balanced_undersample".to_string() }
fn default_max_features() -> usize { 8000 }
fn default_char_ngram_range() -> Vec<usize> { vec![2, 3] }
fn default_word_ngram_range() -> Vec<usize> { vec![1, 2] }
fn default_bow_model_type() -> String { "sgd_logistic".to_string() }
fn default_batch_size() -> usize { 2000 }
fn default_max_seconds() -> usize { 300 }
fn default_max_db_items() -> usize { 100_000 }
fn default_vocab_buckets() -> Vec<VocabBucket> {
    vec![
        VocabBucket {
            name: "high_freq".to_string(),
            min_doc_ratio: 0.05,
            max_doc_ratio: 0.6,
            limit: 1200,
        },
        VocabBucket {
            name: "mid_freq".to_string(),
            min_doc_ratio: 0.01,
            max_doc_ratio: 0.05,
            limit: 2600,
        },
        VocabBucket {
            name: "low_freq".to_string(),
            min_doc_ratio: 0.002,
            max_doc_ratio: 0.01,
            limit: 1200,
        },
    ]
}
fn default_bucket_name() -> String { "bucket".to_string() }
fn default_bucket_limit() -> usize { 1000 }
fn default_one() -> f64 { 1.0 }
fn default_tiktoken_model() -> String { "cl100k_base".to_string() }
fn default_fasttext_dim() -> usize { 64 }
fn default_fasttext_lr() -> f64 { 0.1 }
fn default_fasttext_epoch() -> usize { 5 }
fn default_fasttext_word_ngrams() -> usize { 2 }
fn default_fasttext_minn() -> usize { 2 }
fn default_fasttext_maxn() -> usize { 4 }
fn default_fasttext_bucket() -> usize { 200_000 }
fn default_hashlinear_analyzer() -> String { "char".to_string() }
fn default_hashlinear_ngram_range() -> Vec<usize> { vec![2, 4] }
fn default_hashlinear_n_features() -> usize { 1_048_576 }
fn default_hashlinear_norm() -> Option<String> { Some("l2".to_string()) }
fn default_hashlinear_alpha() -> f64 { 1e-5 }
fn default_hashlinear_epochs() -> usize { 3 }
fn default_hashlinear_batch_size() -> usize { 2048 }
fn default_local_model_type() -> String { "bow".to_string() }
fn default_true() -> bool { true }
