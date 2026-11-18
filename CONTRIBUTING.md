# Contributing to PathSnap

Welcome to the **PathSnap** project repository. This guide defines the collaboration standards, branching models, commit message formats, and quality gates for our 4-member hackathon team.

---

## 1. Branching Strategy

We enforce a structured Git Feature-Branch workflow based on trunk-based development. Direct commits to `main` are strictly forbidden.

### Branch Taxonomy

| Branch Type | Naming Convention | Purpose | Example |
| :--- | :--- | :--- | :--- |
| **Main** | `main` | Production-ready, validated code. Every merge must pass tests. | `main` |
| **Feature** | `feature/<feature-name>` | New pipeline capabilities, preprocessing, or API work. | `feature/candidate-generation` |
| **Experiment**| `exp/<model-family>` | Independent model training or hyperparameter search. | `exp/lightgbm-classifier` |
| **Decoding** | `decoding/<algorithm>` | HMM or Viterbi routing experiments. | `decoding/viterbi-transition` |
| **Bug Fix** | `fix/<bug-description>` | Corrections to existing pipelines or configurations. | `fix/coordinate-projection` |
| **Docs** | `docs/<section>` | Architectural documentation, reports, or guides. | `docs/architecture-update` |

---

## 2. Commit Message Conventions

We adhere to the **Conventional Commits** specification (`<type>(<scope>): <short description>`).

### Allowed Types
- `feat`: A new feature or pipeline capability (e.g., candidate filtering, FastAPI endpoint).
- `fix`: A bug fix or numeric correction (e.g., projection calculation fix).
- `docs`: Documentation changes only (e.g., architecture.md, README updates).
- `style`: Changes that do not affect the meaning of the code (formatting, linting).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `exp`: Machine learning experiments, model fitting runs, or hyperparameter trials.
- `test`: Adding missing tests or correcting existing tests.
- `chore`: Maintenance tasks, repo tooling, or `.gitignore` adjustments.

### Examples
```text
feat(features): implement spatial perpendicular distance and heading diff
exp(models): train baseline random forest on 80-trip training set
refactor(decoding): vectorize viterbi transition matrix computation
docs(readme): update team responsibility matrix and architecture diagram
test(pipeline): add unit tests for graph candidate arc search
```

---

## 3. Pull Request and Code Review Process

1. **Create Branch**: Always branch off the latest `main` branch (`git checkout main && git pull origin main && git checkout -b feature/xyz`).
2. **Local Verification**: Ensure all code adheres to PEP 8 standards and existing unit tests pass before opening a PR.
3. **Pull Request Description**: Detail:
   - What changes were introduced.
   - Which module or PDF guideline requirement is addressed.
   - Instructions on how teammates can reproduce or test the change.
4. **Peer Review**: At least one other team member must review and approve the PR before merging.
5. **Merge Strategy**: Use **Squash and Merge** or **Rebase and Merge** to maintain a linear, bisectable commit history on `main`.

---

## 4. Academic Integrity, Data Safety & Secrets Policy

### Strict Rules:
1. **Never Commit Raw or Processed Data**: All `.track`, `.arcs`, `.nodes`, and `.route` files must remain strictly in local directories excluded by `.gitignore`.
2. **Never Commit Evaluator Ground Truth**: Evaluator route files must never be exposed or pushed to GitHub.
3. **Never Commit Credentials**: API secrets, private SSH keys, and `.env` files must remain local. Use `.env.example` as a template.
4. **No Artificial or Fabricated Results**: Never commit synthetic or hardcoded evaluation scores. All reported results must originate from verifiable code execution.
5. **No AI Copy-Paste of Solutions**: Algorithms must be independently written, understood, and tested by team members.

---

## 5. Coding Standards & Tooling

- **Language**: Python 3.10+
- **Style Guide**: PEP 8 compliance enforced via `black` and `ruff`.
- **Type Hints**: Encourage typing annotations for all public module functions:
  ```python
  def compute_spatial_offset(lat: float, lon: float, arc_geometry: object) -> float:
      ...
  ```
- **Docstrings**: Google or NumPy style docstrings for classes and non-trivial functions.
- **Unit Testing**: Place test files under `tests/test_<module_name>.py`. Run with `pytest`.
