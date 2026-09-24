import subprocess
from datetime import datetime

experiments = [
    # belt
    {"domain_shift_op": "spd1000", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1100", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1200", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1300", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1400", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1500", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1600", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1700", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1800", "anomaly_label": "belt"},
    {"domain_shift_op": "spd1900", "anomaly_label": "belt"},
    {"domain_shift_op": "spd2000", "anomaly_label": "belt"},
    {"domain_shift_op": "spd2400", "anomaly_label": "belt"},
    {"domain_shift_op": "spd2800", "anomaly_label": "belt"},
    {"domain_shift_op": "spd3000", "anomaly_label": "belt"},
    
    # magnetic
    {"domain_shift_op": "spd1000", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1100", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1200", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1300", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1400", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1500", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1600", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1700", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1800", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd1900", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd2000", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd2400", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd2800", "anomaly_label": "magnetic"},
    {"domain_shift_op": "spd3000", "anomaly_label": "magnetic"},

    # normal
    {"domain_shift_op": "spd1000", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1100", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1200", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1300", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1400", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1500", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1600", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1700", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1800", "anomaly_label": "normal"},
    {"domain_shift_op": "spd1900", "anomaly_label": "normal"},
    {"domain_shift_op": "spd2000", "anomaly_label": "normal"},
    {"domain_shift_op": "spd2400", "anomaly_label": "normal"},
    {"domain_shift_op": "spd2800", "anomaly_label": "normal"},
    {"domain_shift_op": "spd3000", "anomaly_label": "normal"}
]

total_experiments = len(experiments)

for i, exp in enumerate(experiments, start=1):
    domain = exp["domain_shift_op"]
    anomaly = exp["anomaly_label"]
    run_name = f"{domain}_{anomaly}"

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando Experiência {i}/{total_experiments}: {run_name}")
    print(f"Parâmetros: {exp}")
    print(f"{'='*60}\n")

    params_list = []
    for key, value in exp.items():
        params_list.append(f"training.{key}={value}")
        
    params_list.append(f"mlflow_run_name={run_name}")

    params_str = ",".join(params_list)

    command = [
        "uv",
        "run",
        "kedro",
        "run",
        "--pipeline=training",
        "--params",
        params_str,
    ]

    result = subprocess.run(command)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All experiments done!")