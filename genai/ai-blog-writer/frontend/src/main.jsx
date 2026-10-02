import {mount,React} from '@rishabh/portfolio-ui';
import {Fields} from '@rishabh/portfolio-ui/fields';
const fields = [{"key": "retrieval", "label": "Retrieval backend", "options": ["tfidf", "chroma", "pinecone"], "default": "tfidf"}, {"key": "words", "label": "Word target", "type": "number", "min": 100, "max": 2000, "default": 500}, {"key": "tone", "label": "Tone", "options": ["professional", "casual", "educational"], "default": "professional"}];
function Controls(props) { return <Fields {...props} fields={fields}/>; }
mount(document.getElementById('root'), {expectedProject: 'ai-blog-writer',Controls});
