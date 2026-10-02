import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "question", "label": "Ask about this episode"}, {"key": "retrieval", "label": "Retrieval backend", "options": ["tfidf", "chroma"], "default": "tfidf"}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'podcast-summarizer',Controls});
