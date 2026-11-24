"""Result management package for GeneWeaver.

Aggregates detected off-target hits, deduplicates candidates across chunk halo boundaries,
and serializes results to CSV and JSON formats.
"""

from geneweaver.result_manager.exporter import ResultExporter

__all__ = ["ResultExporter"]
