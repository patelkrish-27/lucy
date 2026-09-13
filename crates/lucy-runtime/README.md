# Lucy runtime verification loop

The computer-operation runtime executes state-changing HyprFast commands through the cheap command compiler, then schedules a read-only verification observation before allowing the main controller to declare the operation complete. The main model remains responsible for interpreting the observation and recovering when the observed state is wrong.
