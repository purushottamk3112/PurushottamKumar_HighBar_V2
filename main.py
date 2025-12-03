#!/usr/bin/env python3
"""
Kasparro V2 Production System
Main entry point – CLI wrapper for Orchestrator V2
"""

import sys
import argparse
from pathlib import Path

from orchestrator import Orchestrator


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Kasparro V2 – Facebook Ads Performance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --csv data/myfile.csv
  python main.py --config custom.yaml
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file (default from config.yaml if not provided)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()

    print("=" * 80)
    print(" KASPARRO V2 – PRODUCTION SYSTEM ")
    print(" Facebook Ads Performance Analysis Pipeline")
    print("=" * 80)
    print()

    # ------------------------------------------------------------
    # Validate config file
    # ------------------------------------------------------------
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"✗ ERROR: Config file not found: {cfg_path}")
        return 1

    # ------------------------------------------------------------
    # Validate CSV if passed
    # ------------------------------------------------------------
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"✗ ERROR: CSV file not found: {csv_path}")
            print("Tip: Provide a valid file using:")
            print("  python main.py --csv path/to/your.csv")
            return 1

    try:
        # Initialize orchestrator
        orchestrator = Orchestrator(args.config)

        # Run analysis
        result = orchestrator.run(args.csv)

        print()
        print("=" * 80)

        if result.get("success", False):

            print("✓ ANALYSIS COMPLETE")
            print("=" * 80)
            print()

            run_dir = str(result["run_dir"])
            print(f"Run ID: {result['run_id']}")
            print(f"Output Directory: {run_dir}")
            print()
            print("Results:")
            print(f"  - Insights Generated: {result.get('insights_generated', 0)}")
            print(f"  - Insights Validated: {result.get('insights_validated', 0)}")
            print(f"  - Recommendations: {result.get('recommendations_generated', 0)}")
            print()
            print("Output Files:")
            print(f"  - {run_dir}/reports/insights.json")
            print(f"  - {run_dir}/reports/recommendations.json")
            print(f"  - {run_dir}/reports/report.md")
            print(f"  - {run_dir}/logs/*.log")
            print()

            return 0

        else:
            print("✗ ANALYSIS FAILED")
            print("=" * 80)
            print()

            errs = result.get("errors", [])
            if not errs:
                print("No error reason provided.")
            else:
                print("Errors:")
                for e in errs:
                    print(f"  - {e}")

            print()
            return 1

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 130

    except Exception as e:
        print(f"\n✗ Fatal Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
