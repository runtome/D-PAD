import subprocess

# Define parameters
seq_len = 336
dp_values = [0, 0.25, 0.5]
pred_len_values = [96, 192, 336, 720]

# Loop through parameters
for dp in dp_values:
    for pred_len in pred_len_values:
        model_name = f"DPAD_SE_weather_I{seq_len}_o{pred_len}_lr1e-4_bs64_dp{dp}_h336_l2"
        print(f"Running: weather_{seq_len}-{pred_len}")
        
        # Build the command
        command = [
            "python", "run_long.py",
            "--data", "weather",
            "--data_path", "weather.csv",
            "--features", "M",
            "--seq_len", str(seq_len),
            "--pred_len", str(pred_len),
            "--enc_hidden", "336",
            "--levels", "2",
            "--lr", "1e-4",
            "--dropout", str(dp),
            "--batch_size", "64",
            "--RIN", "1",
            "--model_name", model_name,
            "--model", "DPAD_SE"
        ]
        
        # Run the command
        subprocess.run(command)
