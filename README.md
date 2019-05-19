# War
Engine for competitive search of estimators.

The War engine aims at solving a problem, such as maximizing the Gini score for binary classification. The engine works with several simultaneous solution-seeking strategies that generate candidates to solve the problem. Each candidate should be a unique recipe for the solution proposal.

When needed by the engine, the selected strategy generates a candidate encapsulated into a task, which is sent to a work process to evaluate the candidate performance with a validation method such as cross-validation.

Strategies that generate better candidates will be selected the most. To avoid the early domination of lightweight candidates, the engine leases each strategy a warm-up span of 20 candidate solutions for better chances of sustaining the engine's interest.

Given a dataset, a validation method, and the maximum number of CPU processors, namely processing slots, War manages the machine's resources so that the better the strategy is at giving candidates, the more resources it gets. The optional cooperative mode minimizes resource concurrency to maximize slots parallelism.

Solution candidates' results are stored in a Git-like database, which allows the engine caching solutions between sessions and avoids re-evaluating the same candidate solution twice. Such a database may be processed later to extract the overtime performance of each strategy and collect all results into a single place for a more in-depth inspection of each candidate.

## Flowchart of processes

            ================                    ================
             Engine Process                      Worker Process
            ================                    ================
                    │                                   │
                    V                                   V
              Start Workers            ╭────>Wait for an Task from the
                    │                  │     Tasks queue
                    V                  │                │
    ╭────>Generate tasks to fill       │                V
    │     missing slots and put        │           Run the Task
    │     into the Tasks queue to      │                │
    │     be consumed by worker        │                V
    │     processes.                   │     Save result to Database
    │               │                  │                │
    │               V                  │                V
    │     Run UI Dashboard for a       │       Add result to the
    │     few iterations.              │       Results queue
    │               │                  │                │
    │               V                  ╰────────────────╯
    │     Collect all results that
    │     are available in the
    │     Results queue.
    │               │
    ╰───────────────╯

## Scripts

- `work.py`: run a pool of workers to search for candidates.
- `cat.py`: print contents of an object `Strategy/sha256_id`.
- `summary.py`: print summary of the best candidates so far.
- `collect.py`: collect all results into a single file, such as an excel worksheet.
- `plot_strategies.py`: plot the strategies' history in monotonic best performance.
