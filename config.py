MODEL_NAME    = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

KAGGLE_DATASET  = "bishals098/nasa-cmapss-2-engine-degradation"
KAGGLE_USERNAME = "engy999"
KAGGLE_KEY      = "a5bbbbf545d937b16a01d452696623cc"    
DATA_DIR        = r"E:\petromind\data"

HDF5_FILES = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS04.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS06.h5",
    "N-CMAPSS_DS07.h5",
    "N-CMAPSS_DS08a-009.h5",
    "N-CMAPSS_DS08c-008.h5",
    "N-CMAPSS_DS08d-010.h5",
]

#Chunking
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50

#Sampling 
CYCLE_SAMPLE_EVERY = 1

#PostgreSQL
PG_CONN_STR = "postgresql://postgres:gooooooood@localhost:5433/petromind"
WO_TABLE    = "work_orders_rag"

#Retrieval
TOP_K = 5

#LLM 
HF_BASE_URL    = "https://router.huggingface.co/v1"
HF_API_KEY     = "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"        
LLM_MODEL      = "Qwen/Qwen3-Coder-Next:novita"
LLM_TEMP       = 0.2
LLM_MAX_TOKENS = 1024

# N-CMAPSS columns 
SCENARIO_COLS = ["alt", "Mach", "TRA", "T2"]

SENSOR_COLS = [
    "Wf", "Nf", "Nc",
    "T24", "T30", "T48", "T50",
    "P15", "P21", "P24", "Ps30", "P40", "P50",
]

HEALTH_COLS = [
    "Fan_eff_mod", "Fan_flow_mod",
    "LPC_eff_mod", "LPC_flow_mod",
    "HPC_eff_mod", "HPC_flow_mod",
    "HPT_eff_mod", "HPT_flow_mod",
    "LPT_eff_mod", "LPT_flow_mod",
]

UNIT_FAILURE_MODE = {
    # Dev units (real IDs found in DS01-005.h5)
    1:  "LPT_efficiency_flow_HPT_combined",
    2:  "HPT_efficiency_degradation",
    3:  "LPT_efficiency_flow_HPT_combined",
    4:  "LPT_efficiency_flow_HPT_combined",
    5:  "HPT_efficiency_degradation",
    6:  "LPT_efficiency_flow_HPT_combined",
    # Test units
    7:  "HPT_LPT_complex",
    8:  "HPT_LPT_complex",
    9:  "HPT_LPT_complex",
    10: "HPT_LPT_complex",
}

ALERT_ZONE_CYCLES = 50