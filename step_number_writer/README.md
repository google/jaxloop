# Step number writer

The step number writer is an action in Jaxloop that enables writing
the step number out at the end of each inner loop. The interface that
the action uses is `StepNumberWriter.write()`. A sample implementation
of this interface is `LocalFileStepNumberWriter`; this implementation
writes to a local file that is supplied to its constructor.
