import { Meta } from '../layout/Meta';
import { AppConfig } from '../utils/AppConfig';
import { Footer } from './Footer';
import { Form } from './Form';
import { Hero } from './Hero';

const Base1 = () => (
  <div className="antialiased text-gray-600">
    <Meta title={AppConfig.title} description={AppConfig.description} />
    <Hero />
    <Form />

    <Footer />
  </div>
);

export { Base1 };
