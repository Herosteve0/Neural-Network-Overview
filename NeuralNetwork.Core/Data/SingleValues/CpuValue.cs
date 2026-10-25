namespace NeuralNetwork.Core.Data.SingleValues;

public class CpuValue<T> : ISingleValue<T> where T : unmanaged{
    /**
     * <summary>
     * Direct access to the CPU memory of the value.
     * Meant to be used by the Backend code.
     * </summary>
     */
    public T Value { get; set; }

    /**
     * <summary>
     * Create a value from existing data.
     * </summary>
     */
    public CpuValue(T value) {
        Value = value;
    }

    public static implicit operator T(CpuValue<T> value) {
        return value.Value;
    }

    public void Dispose() {}
}