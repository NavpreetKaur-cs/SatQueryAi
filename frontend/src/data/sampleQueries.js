// queryType here is just for reference/documentation of what the model
// should classify each sample as — it's never sent to the backend. The
// model infers the query type itself from the question text.
export const SAMPLE_QUERIES = [
  {
    text: 'How many hectares of cropland are now waterlogged compared to last week\u2019s image?',
    queryType: 'compare',
    requiresCompare: true,
  },
  {
    text: 'Highlight all new construction near this riverbank in the last 6 months.',
    queryType: 'compare',
    requiresCompare: true,
  },
  {
    text: 'Is this SAR image showing a landslide scar or a shadow? Circle it.',
    queryType: 'classify',
    requiresCompare: false,
  },
  {
    text: 'How many built-up structures are visible in this scene?',
    queryType: 'count',
    requiresCompare: false,
  },
];
