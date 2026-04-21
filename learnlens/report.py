"""
learnlens/report.py

print_lqs_report() -- rich terminal output for LQSReport.
Presentation code only. No scoring logic here.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learnlens.scorer import LQSReport


def _bar(value: float, width: int = 10) -> str:
    filled = round(value * width)
    return chr(9608) * filled + chr(9617) * (width - filled)


def _score_color(value: float) -> str:
    if value >= 0.75:
        return "green"
    if value >= 0.50:
        return "yellow"
    return "red"


def print_lqs_report(report: LQSReport) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
        _rich_report(console, report)
    except ImportError:
        from learnlens.scorer import _plain_print_report
        _plain_print_report(report)


def _rich_report(console, report: LQSReport) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    console.print()
    console.rule("[bold cyan]LearnLens Evaluation Report[/bold cyan]")
    console.print(f"[dim]Environment:[/dim] {report.env_url}")
    console.print(f"[dim]Episodes:[/dim]    {report.n_episodes}")
    console.print(f"[dim]Probes:[/dim]      {', '.join(report.probes_run)}")
    console.print(f"[dim]Timestamp:[/dim]   {report.timestamp}")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Metric", style="dim", width=22)
    table.add_column("Score", justify="right", width=6)
    table.add_column("Visual", width=12)
    table.add_column("Notes", width=22)

    table.add_row(
        "Standard Reward",
        f"{report.mean_reward:.2f}",
        _bar(report.mean_reward),
        f"+/- {report.reward_std:.2f} std",
    )
    table.add_section()

    g = report.generalization_score
    table.add_row("Generalization", f"[{_score_color(g)}]{g:.2f}[/]",
                  f"[{_score_color(g)}]{_bar(g)}[/]", "Cross-variant consistency")

    c = report.consistency_score
    table.add_row("Consistency", f"[{_score_color(c)}]{c:.2f}[/]",
                  f"[{_score_color(c)}]{_bar(c)}[/]", "Same state -> same action")

    h = report.hack_index
    h_color = "red" if report.hack_flagged else "green"
    hack_note = "FLAGGED" if report.hack_flagged else "Within tolerance"
    table.add_row("Hack Index", f"[{h_color}]{h:.2f}[/]",
                  f"[{h_color}]{_bar(h)}[/]", f"[{h_color}]{hack_note}[/]")

    r = report.reasoning_score
    r_note = "N/A (disabled)" if "reasoning" not in report.probes_run else "CoT quality"
    table.add_row("Reasoning Quality", f"[{_score_color(r)}]{r:.2f}[/]",
                  f"[{_score_color(r)}]{_bar(r)}[/]", r_note)

    table.add_section()
    rl = report.raw_learning
    table.add_row("  Raw Learning", f"[dim]{rl:.2f}[/dim]",
                  f"[dim]{_bar(rl)}[/dim]", "sqrt(G x C)")
    tc = report.trust_coefficient
    table.add_row("  Trust Coeff", f"[dim]{tc:.2f}[/dim]",
                  f"[dim]{_bar(tc)}[/dim]", "1 - sqrt(H)")

    table.add_section()
    lqs_color = _score_color(report.lqs)
    table.add_row("[bold]LQS (Learning)[/bold]",
                  f"[bold {lqs_color}]{report.lqs:.2f}[/]",
                  f"[bold {lqs_color}]{_bar(report.lqs)}[/]",
                  "[bold]Primary metric[/bold]")

    console.print(table)

    verdict_color = "red" if (report.hack_flagged or report.lqs < 0.4) else \
                   "yellow" if report.lqs < 0.7 else "green"
    console.print(Panel(
        f"[{verdict_color}]{report.verdict()}[/]",
        title="Verdict", border_style=verdict_color,
    ))
    console.print()