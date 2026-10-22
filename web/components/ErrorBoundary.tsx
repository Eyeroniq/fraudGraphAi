"use client";

import { Component, type ReactNode } from "react";
import { ErrorState } from "./ui";

/** Keeps one broken widget from blanking the whole page. */
export class ErrorBoundary extends Component<{ children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };

  static getDerivedStateFromError(error: unknown) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="glass p-6">
          <ErrorState error={this.state.error} retry={() => this.setState({ error: null })} />
        </div>
      );
    }
    return this.props.children;
  }
}
