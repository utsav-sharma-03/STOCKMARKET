import traceback
from pathlib import Path
import pandas as pd
import model_training

log_path = Path(r'd:\fds mini\debug_train_log.txt')
try:
    print('starting')
    result = model_training.train_models()
    print('done', result['best_model_name'])
    log_path.write_text('SUCCESS\n' + str(result['best_model_name']) + '\n', encoding='utf-8')
except Exception as e:
    trace = traceback.format_exc()
    log_path.write_text('ERROR\n' + trace, encoding='utf-8')
    print(trace)
