from __future__ import annotations

from copy import deepcopy
import unittest

from information_upper_bound.unit_sampling import (
    select_resampling_units,
    validate_resampling_unit_selection,
)


def _rows() -> list[dict]:
    return [
        {
            "id": f"{unit_id}-{row_index}",
            "question": f"Question {unit_id}-{row_index}",
            "choices": ["No", "Yes"],
            "answer": "A" if row_index == 0 else "B",
            "diagnostic": {
                "resampling_unit_id": unit_id,
                "oracles": {"ordered_events": [{"text": f"event-{row_index}"}]},
            },
        }
        for unit_id in ("scene-0", "scene-1", "scene-2", "scene-3", "scene-4")
        for row_index in range(2)
    ]


class ResamplingUnitSelectionTests(unittest.TestCase):
    def _select(self, rows: list[dict] | None = None):
        return select_resampling_units(
            rows or _rows(),
            dataset="clevrer",
            canonical_split="validation",
            sample_size=2,
            seed=42,
        )

    def test_golden_selection_is_row_order_independent_and_unit_complete(self) -> None:
        selected, options, report = self._select()
        reversed_selected, reversed_options, reversed_report = self._select(
            list(reversed(_rows()))
        )
        self.assertEqual(report["selected_unit_ids"], ["scene-3", "scene-4"])
        self.assertEqual(options, reversed_options)
        self.assertEqual(report, reversed_report)
        self.assertEqual(
            {row["id"] for row in selected},
            {row["id"] for row in reversed_selected},
        )
        self.assertEqual(len(selected), 4)

    def test_ranking_does_not_read_answers_questions_choices_or_oracles(self) -> None:
        _selected, options, report = self._select()
        mutated = deepcopy(_rows())
        for row in mutated:
            row["question"] = "Completely different"
            row["choices"] = ["changed", "content"]
            row["answer"] = "B" if row["answer"] == "A" else "A"
            row["diagnostic"]["oracles"] = {"operator": [{"secret": "changed"}]}
        _mutated_selected, mutated_options, mutated_report = self._select(mutated)
        self.assertEqual(options, mutated_options)
        self.assertEqual(
            report["selected_unit_ids"], mutated_report["selected_unit_ids"]
        )

    def test_invalid_requests_fail_without_silent_clamping(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            select_resampling_units(
                _rows(),
                dataset="clevrer",
                canonical_split="validation",
                sample_size=0,
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            select_resampling_units(
                _rows(),
                dataset="clevrer",
                canonical_split="validation",
                sample_size=6,
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "seed"):
            select_resampling_units(
                _rows(),
                dataset="clevrer",
                canonical_split="validation",
                sample_size=2,
                seed=-1,
            )

    def test_selection_report_authenticates_ranking_and_unit_closure(self) -> None:
        selected, options, report = self._select()
        validate_resampling_unit_selection(
            report=report,
            options=options,
            selected_rows=selected,
            dataset="clevrer",
            canonical_split="validation",
        )

        tampered_ids = deepcopy(report)
        tampered_ids["selected_unit_ids"] = ["scene-0", "scene-1"]
        with self.assertRaisesRegex(ValueError, "hash ranking"):
            validate_resampling_unit_selection(
                report=tampered_ids,
                options=options,
                selected_rows=selected,
                dataset="clevrer",
                canonical_split="validation",
            )

        partial_unit = selected[:-1]
        with self.assertRaisesRegex(ValueError, "every record"):
            validate_resampling_unit_selection(
                report=report,
                options=options,
                selected_rows=partial_unit,
                dataset="clevrer",
                canonical_split="validation",
            )


if __name__ == "__main__":
    unittest.main()
