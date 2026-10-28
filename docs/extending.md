# Keeping CMRBench extensible

The framework boundary is dataset → sample → sampling → reconstruction input →
model → result → evaluation. Preserve these contracts when adding a component.

For a new dataset, subclass `CineMRIDataset`, return `CineMRISample`, and register
a factory with `DatasetRegistry`. Keep physical acquisition masks when present.
Define reference meaning, complex phase handling, axes and selection explicitly;
do not put model-specific normalization in the reader.

For a new model, implement `ReconstructionModel`, register a lazy factory with
`ModelRegistry`, and return the candidate plus transform/provenance records.
Perform private tensor conversions inside the adapter. Validate supported
conditioning and calibration before inference. For a new sampler, implement
`SamplingProtocol` and return `SamplingResult` with the realized mask, actual R,
ACS policy and implementation provenance. Register different source algorithms
as explicit implementations rather than overloading a name based on the model.
