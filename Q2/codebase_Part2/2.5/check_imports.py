
import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'Jonschkowski (18)'))

try:
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    print("TensorFlow V1 compat enabled.")
    
    try:
        import sonnet as snt
        print(f"Sonnet imported: {snt.__version__ if hasattr(snt, '__version__') else 'unknown'}")
    except ImportError:
        print("Sonnet not found.")

    from methods.dpf import DPF
    print("DPF class imported successfully.")

except Exception as e:
    print(f"Import failed: {e}")
