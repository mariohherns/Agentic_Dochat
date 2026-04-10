import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Props = {
  content: string;
};

export function MarkdownBlock({ content }: Props) {
  return (
    <div className="mt-2 text-sm text-[rgb(var(--fg)/0.9)]">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mb-3 mt-4 text-lg font-semibold first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mb-2 mt-4 text-base font-semibold first:mt-0">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="mb-2 mt-3 text-sm font-semibold first:mt-0">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="mb-3 last:mb-0 leading-6 text-[rgb(var(--fg)/0.9)]">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="mb-3 list-disc space-y-1 pl-5 marker:text-[rgb(var(--fg)/0.65)]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="mb-3 list-decimal space-y-1 pl-5 marker:text-[rgb(var(--fg)/0.65)]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-6 text-[rgb(var(--fg)/0.9)]">{children}</li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-[rgb(var(--fg))]">
              {children}
            </strong>
          ),
          em: ({ children }) => <em className="italic">{children}</em>,
          code: ({ children }) => (
            <code className="rounded bg-[rgb(var(--card-muted))] px-1.5 py-0.5 text-xs ring-1 ring-inset ring-[rgb(var(--border))]">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="mb-3 overflow-x-auto rounded-md bg-[rgb(var(--card-muted))] p-3 text-xs ring-1 ring-inset ring-[rgb(var(--border))]">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="mb-3 overflow-x-auto">
              <table className="w-full border-collapse text-left text-sm ring-1 ring-inset ring-[rgb(var(--border))]">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[rgb(var(--card-muted))]">{children}</thead>
          ),
          th: ({ children }) => (
            <th className="border border-[rgb(var(--border))] px-3 py-2 font-semibold text-[rgb(var(--fg))]">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-[rgb(var(--border))] px-3 py-2 align-top text-[rgb(var(--fg)/0.9)]">
              {children}
            </td>
          ),
          blockquote: ({ children }) => (
            <blockquote className="mb-3 border-l-2 border-[rgb(var(--border))] pl-4 italic text-[rgb(var(--fg)/0.75)]">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-4 border-[rgb(var(--border))]" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}


export function formatAsList(text?: string | null) {
  if (!text) return "";

  // If already contains markdown list, return as-is
  if (/^\s*[-*]\s+/m.test(text) || /^\s*\d+\.\s+/m.test(text)) {
    return text;
  }

  // Split by new lines and turn into bullet list
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `- ${line}`)
    .join("\n");
}