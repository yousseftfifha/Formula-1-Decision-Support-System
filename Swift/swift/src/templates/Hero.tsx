import Link from 'next/link';

import { Background } from '../background/Background';
import { HeroOneButton } from '../hero/HeroOneButton';
import { Section } from '../layout/Section';
import { NavbarTwoColumns } from '../navigation/NavbarTwoColumns';

const Hero = () => (
  <Background color="bg-gray-100">
    <Section yPadding="py-6">
      <NavbarTwoColumns>
        <li>
          <Link href="/">
            <a>Dashboard</a>
          </Link>
        </li>
        <li>
          <Link href="/indexForm">
            <a>Form</a>
          </Link>
        </li>
        <li>
          <Link href="/">
            <a>About</a>
          </Link>
        </li>
        <li>
          <Link href="/">
            <a>Contact</a>
          </Link>
        </li>
      </NavbarTwoColumns>
    </Section>

    <Section yPadding="pt-1 pb-2">
      <HeroOneButton
        title={
          <>
            {'The modern dashboard  for\n'}
            <span className="text-primary-500">Formula One Investors</span>
          </>
        }
      />
    </Section>
  </Background>
);

export { Hero };
